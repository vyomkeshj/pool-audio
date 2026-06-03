#!/usr/bin/env python3
"""Pool-Audio desktop app — two independent halves:

  LEFT  · PLAYER    pick an operating condition (suction/discharge blockage level,
                    aeration on/off, which machines, sensor, noise); the app finds a
                    matching recording in the dataset and plays it on the speakers.

  RIGHT · LISTENER  a separate engine that captures audio — the system-audio loopback
                    (what's playing) or the microphone — and, knowing nothing about
                    the player, reports which machines are running, aeration on/off,
                    and the discharge & suction blockage levels.

The listener uses an 8 s window for aeration/which-machine (their models are trained
on 8 s) and a rolling buffer up to 60 s for blockage (those models need a long
window). Blockage shows "warming up / ~ / reliable" as the buffer fills.

Run:  ./run/run_gui.sh   (or  python3 run/app_desktop.py)
"""
import os, sys, threading, queue, subprocess
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal as ssignal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from clip_library import ClipLibrary
from listener import (Listener, stream_capture, stream_file, default_monitor,
                      WIN_SHORT_S, SR)

BG = "#1e1f26"; CARD = "#2a2c36"; FG = "#e8e8ee"; MUTED = "#9aa0b5"
ACC = "#4da3ff"; ON = "#46d17a"; OFF = "#3a3d4a"; WARN = "#f0b54b"


class App:
    def __init__(self, root):
        self.root = root
        root.title("Pool-Audio — player & independent listener")
        root.configure(bg=BG)
        root.geometry("1120x800")
        root.minsize(940, 680)
        self.lib = ClipLibrary()
        self.listener = Listener("auto")
        self.play_proc = None
        self.listen_stop = threading.Event()
        self.listen_flush = threading.Event()
        self.listen_thread = None
        self.q = queue.Queue()
        self.current_clip = None
        self._last_audio = None
        self._poll_n = 0

        self._style()
        wrap = tk.Frame(root, bg=BG); wrap.pack(fill="both", expand=True, padx=12, pady=12)
        wrap.columnconfigure(0, weight=1, uniform="x")
        wrap.columnconfigure(1, weight=1, uniform="x")
        wrap.rowconfigure(0, weight=3)
        wrap.rowconfigure(1, weight=2)
        self._build_player(wrap)
        self._build_listener(wrap)
        self._build_spectrogram(wrap)
        self.root.after(120, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")          # clam honours combobox colours reliably
        except tk.TclError:
            s.theme_use("default")
        FIELD = "#3a3d4a"
        s.configure("TCombobox", fieldbackground=FIELD, background=FIELD,
                    foreground=FG, arrowcolor=FG, bordercolor=FIELD,
                    lightcolor=FIELD, darkcolor=FIELD, selectbackground=FIELD,
                    selectforeground=FG, padding=3)
        # readonly is the state our comboboxes are in — force readable colours
        s.map("TCombobox",
              fieldbackground=[("readonly", FIELD), ("disabled", CARD)],
              foreground=[("readonly", FG), ("disabled", MUTED)],
              selectbackground=[("readonly", FIELD)],
              selectforeground=[("readonly", FG)],
              arrowcolor=[("readonly", FG)])
        # the drop-down popup list (a Tk Listbox under the hood)
        self.root.option_add("*TCombobox*Listbox.background", FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACC)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#11121a")
        s.configure("Card.TFrame", background=CARD)
        s.configure("TCheckbutton", background=CARD, foreground=FG)
        s.map("TCheckbutton", background=[("active", CARD)],
              foreground=[("active", FG)])

    # ---------------------------------------------------------------- PLAYER
    def _build_player(self, parent):
        card = tk.Frame(parent, bg=CARD); card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(card, text="▶  PLAYER", bg=CARD, fg=ACC,
                 font=("DejaVu Sans", 14, "bold")).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(card, text="Pick a condition; the app plays a matching recording.",
                 bg=CARD, fg=MUTED, font=("DejaVu Sans", 9)).pack(anchor="w", padx=16)

        form = tk.Frame(card, bg=CARD); form.pack(fill="x", padx=16, pady=12)
        self.pv = {}
        def row(r, label, key, values, default):
            tk.Label(form, text=label, bg=CARD, fg=FG, width=16, anchor="w").grid(
                row=r, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=str(default)); self.pv[key] = v
            ttk.Combobox(form, textvariable=v, values=[str(x) for x in values],
                         state="readonly", width=14).grid(row=r, column=1, sticky="w")
        row(0, "Sensor type", "dev_type", ["mic", "cam"], "cam")
        row(1, "Discharge level", "valveOut", ["any"] + self.lib.options("valveOut"), 1)
        row(2, "Suction level", "valveIn", ["any"] + self.lib.options("valveIn"), 1)
        row(3, "Aeration", "aeration", ["off", "on"], "off")
        row(4, "Noise", "noise_cat", ["any", "N", "A", "B", "C", "D", "E"], "N")

        mfr = tk.Frame(card, bg=CARD); mfr.pack(fill="x", padx=16)
        tk.Label(mfr, text="Also running:", bg=CARD, fg=FG).pack(side="left")
        self.mv = {}
        for m, lab in [("M2", "M2 pump"), ("M3", "M3 fan"), ("M4", "M4 fan")]:
            var = tk.IntVar(value=0); self.mv[m] = var
            ttk.Checkbutton(mfr, text=lab, variable=var).pack(side="left", padx=6)

        btns = tk.Frame(card, bg=CARD); btns.pack(fill="x", padx=16, pady=14)
        self._mkbtn(btns, "Find & Play", self.find_and_play, ACC).pack(side="left")
        self._mkbtn(btns, "Stop", self.stop_play, OFF).pack(side="left", padx=8)

        self.play_status = tk.Label(card, text="No clip loaded.", bg=CARD, fg=MUTED,
                                    font=("DejaVu Sans", 9), justify="left", wraplength=420)
        self.play_status.pack(anchor="w", padx=16, pady=(0, 4))
        self.truth = tk.Label(card, text="", bg=CARD, fg=ON, justify="left",
                              font=("DejaVu Sans Mono", 9), wraplength=440)
        self.truth.pack(anchor="w", padx=16, pady=(0, 12))

    def _mkbtn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd, bg=color, fg="#11121a",
                         activebackground=color, relief="flat", padx=14, pady=6,
                         font=("DejaVu Sans", 10, "bold"), cursor="hand2")

    def _sel(self, key):
        v = self.pv[key].get()
        if v in ("any", ""):
            return None
        if key == "aeration":
            return 1 if v == "on" else 0
        if key in ("dev_type", "noise_cat"):
            return v
        return int(v)

    def find_and_play(self):
        clip = self.lib.find(
            dev_type=self._sel("dev_type"), valveOut=self._sel("valveOut"),
            valveIn=self._sel("valveIn"), aeration=self._sel("aeration"),
            noise_cat=self._sel("noise_cat"),
            M2=self.mv["M2"].get(), M3=self.mv["M3"].get(), M4=self.mv["M4"].get())
        if clip is None:
            self.play_status.config(text="No recording matches that combination.", fg=WARN)
            self.truth.config(text="")
            return
        self.current_clip = clip
        self.stop_play()
        self.listen_flush.set()      # tell the listener to start a fresh buffer
        self._last_audio = None
        self.play_proc = subprocess.Popen(["aplay", "-q", clip["path"]])
        self.play_status.config(text=f"Playing: {clip['file']}", fg=FG)
        self.truth.config(text="ground truth ▸ " + ClipLibrary.describe(clip))

    def stop_play(self):
        if self.play_proc and self.play_proc.poll() is None:
            self.play_proc.terminate()
        self.play_proc = None

    # -------------------------------------------------------------- LISTENER
    def _build_listener(self, parent):
        card = tk.Frame(parent, bg=CARD); card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        tk.Label(card, text="👂  LISTENER  (independent)", bg=CARD, fg=ACC,
                 font=("DejaVu Sans", 14, "bold")).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(card, text="Captures audio and infers the condition — blind to the player.",
                 bg=CARD, fg=MUTED, font=("DejaVu Sans", 9)).pack(anchor="w", padx=16)

        ctl = tk.Frame(card, bg=CARD); ctl.pack(fill="x", padx=16, pady=12)
        tk.Label(ctl, text="Source", bg=CARD, fg=FG).grid(row=0, column=0, sticky="w")
        self.src = tk.StringVar(value="loopback — hears what's playing")
        ttk.Combobox(ctl, textvariable=self.src, state="readonly", width=26,
                     values=["loopback — hears what's playing",
                             "microphone — room sound"]).grid(row=0, column=1, padx=6)
        # sensor family is auto-detected from the audio — no manual selector
        self.sensor_lbl = tk.Label(ctl, text="sensor: (auto)", bg=CARD, fg=MUTED,
                                   font=("DejaVu Sans", 9))
        self.sensor_lbl.grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.listen_btn = self._mkbtn(ctl, "Start listening", self.toggle_listen, ON)
        self.listen_btn.grid(row=0, column=3, padx=(12, 0))

        # readouts
        body = tk.Frame(card, bg=CARD); body.pack(fill="both", expand=True, padx=16, pady=6)
        self.r_running = self._readout(body, "Running machines")
        self.r_aer = self._readout(body, "Aeration")
        self.r_dis = self._readout(body, "Discharge blockage")
        self.r_suc = self._readout(body, "Suction blockage")
        self.buf_lbl = tk.Label(card, text="idle", bg=CARD, fg=MUTED,
                                font=("DejaVu Sans Mono", 9))
        self.buf_lbl.pack(anchor="w", padx=16, pady=(0, 12))

    # ----------------------------------------------------------- SPECTROGRAM
    def _build_spectrogram(self, parent):
        card = tk.Frame(parent, bg=CARD)
        card.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
        tk.Label(card, text="〰  LIVE SPECTROGRAM  (captured audio)", bg=CARD, fg=ACC,
                 font=("DejaVu Sans", 11, "bold")).pack(anchor="w", padx=16, pady=(8, 0))
        self.fig = Figure(figsize=(9, 2.2), facecolor=CARD)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#15161c")
        self.fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.18)
        self.spec_im = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=6)
        self._spec_clear()

    def _spec_clear(self):
        self.ax.clear()
        self.ax.set_facecolor("#15161c")
        self.ax.set_ylabel("Hz", color=MUTED, fontsize=8)
        self.ax.set_xlabel("time (s)", color=MUTED, fontsize=8)
        self.ax.tick_params(colors=MUTED, labelsize=7)
        self.ax.text(0.5, 0.5, "start listening to see the spectrogram",
                     ha="center", va="center", color=MUTED, transform=self.ax.transAxes)
        self.spec_im = None
        self.canvas.draw_idle()

    def _spec_update(self, audio):
        n = min(len(audio), int(12 * SR))
        x = audio[-n:]
        rms_db = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
        self.ax.clear()
        if rms_db < -70:                       # silent input → say so, don't fake it
            self.ax.set_facecolor("#15161c")
            self.ax.text(0.5, 0.5, "no audio on this input (%.0f dB)\n"
                         "start playback, or switch Source to loopback" % rms_db,
                         ha="center", va="center", color=WARN,
                         transform=self.ax.transAxes, fontsize=10)
        else:
            f, t, Sxx = ssignal.spectrogram(x, SR, nperseg=2048, noverlap=1024)
            keep = f <= 8000
            S = 10 * np.log10(Sxx[keep] + 1e-12)
            vmax = float(S.max())
            self.ax.imshow(S, origin="lower", aspect="auto",
                           extent=[0, len(x) / SR, 0, 8000], cmap="magma",
                           vmin=vmax - 75, vmax=vmax)     # fixed 75 dB range
        self.ax.set_ylabel("Hz", color=MUTED, fontsize=8)
        self.ax.set_xlabel("time (s, last %ds)" % (n // SR), color=MUTED, fontsize=8)
        self.ax.tick_params(colors=MUTED, labelsize=7)
        self.canvas.draw_idle()

    def _readout(self, parent, title):
        fr = tk.Frame(parent, bg=CARD); fr.pack(fill="x", pady=7)
        tk.Label(fr, text=title, bg=CARD, fg=MUTED, width=18, anchor="w",
                 font=("DejaVu Sans", 10)).pack(side="left")
        val = tk.Label(fr, text="—", bg=CARD, fg=FG, anchor="w",
                       font=("DejaVu Sans", 13, "bold"))
        val.pack(side="left", fill="x", expand=True)
        return val

    def toggle_listen(self):
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_stop.set()
            self.listen_btn.config(text="Start listening", bg=ON)
            self._last_audio = None
            self.buf_lbl.config(text="stopped", fg=MUTED)
            return
        self.listener.set_sensor("auto")     # sensor family detected from audio
        source = "mic" if self.src.get().startswith("micro") else "loopback"
        self.listen_stop.clear()
        self.listen_flush.clear()
        self.listen_thread = threading.Thread(target=self._listen_loop,
                                              args=(source,), daemon=True)
        self.listen_thread.start()
        self.listen_btn.config(text="Stop listening", bg=WARN)

    def _listen_loop(self, source):
        try:
            gen = stream_capture(source, WIN_SHORT_S, 2.0, None,
                                 monitor=default_monitor(), flush_event=self.listen_flush)
            for buf, t in gen:
                if self.listen_stop.is_set():
                    break
                self._last_audio = buf
                r = self.listener.analyze(buf)
                self.q.put(r)
        except Exception as e:
            self.q.put({"error": str(e)})

    def _poll(self):
        got = False
        try:
            while True:
                r = self.q.get_nowait()
                self._show(r); got = True
        except queue.Empty:
            pass
        # refresh spectrogram from the latest captured audio (throttled)
        self._poll_n += 1
        if got and self._last_audio is not None and len(self._last_audio) > SR:
            self._spec_update(self._last_audio)
        self.root.after(150, self._poll)

    def _show(self, r):
        if "error" in r:
            self.buf_lbl.config(text=f"capture error: {r['error']}", fg=WARN)
            return
        lvl = r.get("input_rms_db", -120)
        sens = r.get("sensor", "—")
        self.sensor_lbl.config(
            text=f"sensor: {sens} (auto-detected)" if sens not in ("—",) else "sensor: (auto)",
            fg=ACC if sens not in ("—",) else MUTED)
        if r.get("silent"):
            for w in (self.r_running, self.r_aer, self.r_dis, self.r_suc):
                w.config(text="—", fg=MUTED)
            self.r_running.config(text="⚠ no audio on this input", fg=WARN)
            src = "loopback" if not self.src.get().startswith("micro") else "microphone"
            hint = (" — is anything playing?" if src == "loopback"
                    else " — this mic reads silence; try the loopback source")
            self.buf_lbl.config(text=f"input {lvl:.0f} dB (silent){hint}", fg=WARN)
            return
        run = r["running"]; extra = [m for m in run if m != "M1"]
        self.r_running.config(
            text="M1 (always on)" + ("  +  " + ", ".join(extra) if extra else "   (M1 only)"),
            fg=ON if extra else FG)
        a = r["aeration"]
        self.r_aer.config(text=f"{'ON' if a['on'] else 'off'}   (p={a['p']:.2f})",
                          fg=ON if a["on"] else FG)
        b = r.get("blockage", {})
        if "discharge" in b:
            for axis, widget in [("discharge", self.r_dis), ("suction", self.r_suc)]:
                x = b[axis]
                tag = {"reliable": "", "partial": "  (firming…)",
                       "warming_up": "  (warming up)",
                       "uncertain_aux": "  (uncertain · other equipment running)"}[x["reliability"]]
                if x["reliability"] == "uncertain_aux":
                    widget.config(text=f"— {tag.strip()}", fg=MUTED)
                else:
                    txt = f"level {x['level']}{tag}" if x["restricted"] else f"open (L1){tag}"
                    widget.config(text=txt, fg=WARN if x["restricted"] else FG)
        self.buf_lbl.config(
            text=f"buffer {r['buffer_s']:.0f}s · input {lvl:.0f} dB · sensor={r['sensor']}",
            fg=ON if lvl > -55 else MUTED)

    def _on_close(self):
        self.listen_stop.set()
        self.stop_play()
        self.root.after(200, self.root.destroy)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
