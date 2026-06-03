#!/usr/bin/env python3
"""Pool-Audio — výzkumná zpráva (česká verze).

Česká lokalizace reportu z ../run/report_app.py. Veškerá data, modely a vykreslovací
pomocné funkce se sdílejí z anglické verze (žádná duplikace artefaktů); zde se
překládá pouze text.

Spuštění:  ./run_cz/run_report_cz.sh   (nebo streamlit run run_cz/report_app.py)
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as ssignal

# Czech page config FIRST, then import the English module (its set_page_config is
# guarded, so it becomes a no-op and the Czech title wins).
st.set_page_config(page_title="Pool-Audio — dokumentace (CZ)", layout="wide")

RUN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run")
sys.path.insert(0, RUN)
import report_app as R  # noqa: E402  -- shared loaders, plot helpers, models

SR = R.SR
db = R.db
fig_to_st = R.fig_to_st

NOISE_CZ = {"N": "čisté (referenční)", "A": "dětské hřiště", "B": "sekačka",
            "C": "doprava", "D": "lidská řeč", "E": "hudba"}


# --------------------------------------------------------------------- STRÁNKY
def page_overview():
    st.title("🔊 Čtení stavu bazénového čerpadla ze zvuku")
    st.markdown("""
Jediná ~60sekundová nahrávka mikrofonem (nebo mikrofonem kamery) z čerpací stanice
dokáže určit, **co zařízení dělá** — bez fyzického kontaktu. Vše v tomto datasetu je
**zdravé zařízení v různých provozních konfiguracích**; nic není rozbité. Mikrofon
odpovídá na dvě *nezávislé* otázky:
""")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Osa A — omezení průtoku M1")
        st.markdown("""
Hlavní čerpadlo **M1** běží vždy; omezují ho dva škrticí ventily.
- **Sací** strana (`valveIn`, 1–8) — jak je přiškrcený vstup?
- **Výtlačná** strana (`valveOut`, 1–11) — jak je přiškrcený výstup?

`1 = plně otevřeno`, vyšší = více omezeno. To je hlavní cíl monitorování stavu,
měřený za všech typů hluku na pozadí.""")
    with c2:
        st.subheader("Osa B — které zařízení běží")
        st.markdown("""
Které zdravé stroje běží spolu s M1:
- **M2** — druhé velké čerpadlo
- **M3** — odsávací ventilátor (hlučný, proudění ~4–8 kHz)
- **M4** — odsávací ventilátor (téměř neslyšný)
- **aerace** — vzduchový injektor (zap/vyp)""")

    st.divider()
    st.subheader("Hlavní výsledky \\*")
    cols = st.columns(4)
    cols[0].metric("Úroveň výtlaku", "do ±1 = 1,00", "přesně ≈0,98 (kamera)")
    cols[1].metric("Úroveň sání", "do ±1 = 1,00", "MAE ≈0,35 kroku")
    cols[2].metric("Aerace zap/vyp", "AUC 1,00 (kamera)", "F1 0,97–1,00")
    cols[3].metric("Který stroj", "M3 ≈1,0, M2 ≈0,9", "M4 těžké na mic")
    st.caption("13 850 zvukových souborů · 7 nahrávacích kampaní · modely trénované "
               "zvlášť pro každý typ senzoru · v postranním panelu projděte signatury, "
               "modely a živou ukázku.")
    st.divider()
    st.markdown("""
\\* **Jak byly tyto hodnoty změřeny (a proč jim lze věřit).** Každé číslo pochází
z *křížové validace s odděleným testem*, která nikdy nedovolí, aby se nahrávka — nebo
její téměř-duplikát — objevila současně v tréninku i testu. Hodnoty tedy odpovídají
skutečně nové nahrávce, ne zapamatování. Tři opatření:

- **Sourozenecké kanály drženy pohromadě.** Každá nahrávka zachycuje 16 senzorů
  (8 mikrofonů + 8 kamer) ve *stejném okamžiku*; tyto téměř identické sourozence
  nikdy nerozdělíme mezi trénink a test. *(Naivní rozdělení přesnost nadhodnotí —
  např. 0,977 → 0,993.)*
- **Celé konfigurace ventilů oddělené** (včetně všech opakování a zašuměných verzí) —
  model tak nemůže ohodnotit nahrávku podle jejího sourozence se stejným nastavením.
- **Oddělené i celé nahrávací dny a celé úrovně omezení**, aby se ověřilo, že to
  funguje na nový den a že umí úrovně interpolovat.
""")


def page_dataset():
    st.title("📁 Dataset a prostor parametrů")
    idx = R.load_index()
    fs = idx["folder_summary"]
    st.markdown(f"**{idx['_meta']['n_files_total']:,} nahrávek** v **{len(fs)} "
                "kampaních**; každá relace zachycuje 16 zařízení současně "
                "(8 mikrofonů + 8 mikrofonů kamer).")
    st.dataframe(pd.DataFrame([{"kampaň": k, "soubory": v["n_files"]}
                               for k, v in fs.items()]), hide_index=True,
                 width="stretch")

    st.subheader("Společné pokrytí ventilů (relace pouze s M1)")
    st.caption("Sání a výtlak se mění **společně** — většina nahrávek je omezená na "
               "obou stranách — proto modelujeme dvě nezávislé ordinální osy, ne jeden "
               "společný štítek. Buňky = počet nahrávacích relací.")
    df = R.load_features()
    m1 = df[(df.M2 == 0) & (df.M3 == 0) & (df.M4 == 0) & (df.aeration == 0)]
    piv = (m1.drop_duplicates("session").groupby(["valveIn", "valveOut"]).size()
           .unstack(fill_value=0))
    piv.index.name = "valveIn ↓ / valveOut →"
    st.dataframe(piv, width="stretch")

    st.subheader("Sada pro odolnost vůči hluku na pozadí")
    st.markdown("Každá úroveň omezení byla nahrána i pod pěti hluky z prostředí, aby "
                "šlo testovat odolnost modelů:")
    st.markdown("  ·  ".join(f"**{k}** = {v}" for k, v in NOISE_CZ.items()))
    st.info("⚠️ Aerace a pomocné stroje (M2/M3/M4) byly nahrány jen v jedné čisté "
            "kampani (5_25); aerace pouze při jedné konfiguraci ventilů (`vin1/vout1`). "
            "Aerace / který-stroj jsou tedy validovány na čistých datech a *úroveň* "
            "aerace nelze určit — pouze zap/vyp.")


def page_signatures():
    st.title("〰 Jak se mění akustická signatura")
    sigs = R.load_sigs()
    dt = st.radio("Typ senzoru", ["cam", "mic"], horizontal=True,
                  help="Mikrofony a mikrofony kamer jsou akusticky odlišné senzory a "
                       "modelují se zvlášť.")
    st.caption("Průměrná spektra (Welch) přes čisté nahrávky (hluk=N) pro každou "
               "podmínku, normalizovaná na celkový výkon (jde tedy o *tvar* spektra, "
               "ne o hlasitost).")

    st.header("Sání vs výtlak — vypadají jinak")
    st.markdown("""
**Výtlačné** škrcení (výstup) způsobuje *velkou, monotónní* změnu: tón čerpadla
stoupá a s mírou omezení roste energie ve středním pásmu (250–500 Hz) — silný, snadno
čitelný signál. **Sací** škrcení (vstup) je jemnější: hlavně zvedá velmi nízké pásmo
(0–100 Hz) a snižuje tón — čitelné, ale menší efekt (těžší osa).""")
    c1, c2 = st.columns(2)
    with c1:
        fig_to_st(R._psd_sweep(sigs, dt, "discharge", [1, 2, 3, 4, 5, 8, 11],
                               "Výtlak (valveOut) — přehled úrovní", "vout"))
    with c2:
        fig_to_st(R._psd_sweep(sigs, dt, "suction", [1, 2, 3, 4, 5],
                               "Sání (valveIn) — přehled úrovní", "vin"))

    st.header("Aerace zapnutá vs vypnutá")
    st.markdown("""
Zapnutí **aerace** při pevné konfiguraci ventilů přidá nízkofrekvenční / tónovou
strukturu (**zesílení 50–100 Hz** u mikrofonů; tónový/harmonický posun u kamer).
Hlasitost se téměř nemění — rozdíl je ve *tvaru* spektra, proto funguje detektor
nezávislý na hlasitosti (a dřívější historka o „−34 dB artefaktu hlasitosti“ byla
chyba způsobená průměrováním).""")
    fig_to_st(R._psd_pair(sigs, dt, "aer_off", "aer_on", "aerace vyp", "aerace ZAP",
                          "Aerace zap vs vyp (shodné vin1/vout1)"))

    st.header("Který stroj běží")
    st.markdown("""
Každý pomocný stroj přidá svůj otisk. **M3** (hlučný odsávací ventilátor) vnáší
širokopásmové **4–8 kHz** proudění — nezaměnitelné. **M2** (2. čerpadlo) přidá
střední/nízké tónové linky. **M4** (téměř neslyšný ventilátor) spektrum téměř
nezmění — nejhůře slyšitelný.""")
    m = st.selectbox("Stroj", ["M3", "M2", "M4"],
                     format_func=lambda x: {"M3": "M3 — odsávací ventilátor (hlučný)",
                                            "M2": "M2 — 2. velké čerpadlo",
                                            "M4": "M4 — odsávací ventilátor (téměř neslyšný)"}[x])
    fig_to_st(R._psd_pair(sigs, dt, f"{m}_off", f"{m}_on", f"{m} vyp", f"{m} ZAP",
                          f"{m} zap vs vyp (shodné, pouze M1)"))


def page_channels():
    st.title("🎙 8 mikrofonů nezní stejně — rozhoduje umístění")
    st.markdown("""
Každá relace nahrává stejný okamžik na **8 mikrofonů + 8 mikrofonů kamer**. Nejsou
zaměnitelné: kde senzor sedí (a vestavěné AGC u kamer) mění jak to, co zachytí, tak i
to, jak dobře z toho lze přečíst úroveň ucpání. To je největší faktor kvality modelu —
proto modelujeme **zvlášť pro každý typ senzoru** a nikdy nemícháme mikrofony
s kamerami.""")
    psd = R.load_channels_psd(); ch = R.load_channels()
    if not psd or not ch:
        st.warning("Spusťte `python3 run/channels.py` pro vygenerování dat po kanálech.")
        return

    st.header("Spektra po kanálech (stejná provozní podmínka)")
    st.caption("Průměrné spektrum každého kanálu přes čisté nahrávky pouze s M1, "
               "normalizováno na celkový výkon (tvar spektra). Všimněte si, jak "
               "mikrofonní kanály umisťují tón čerpadla různě (≈130–390 Hz), zatímco "
               "mikrofony kamer sedí mnohem výše.")
    f = psd["freq"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6))
    for i in range(1, 9):
        if f"mic{i}" in psd:
            a1.semilogx(f, db(psd[f"mic{i}"]), lw=1.3, label=f"mic{i}")
    a1.set_title("8 mikrofonů"); a1.set_xlim(30, f.max()); a1.set_xlabel("Hz")
    a1.set_ylabel("výkon (dB, rel.)"); a1.legend(ncol=2, fontsize=7); a1.grid(alpha=0.2)
    for i in range(1, 9):
        if f"cam{i}" in psd:
            a2.semilogx(f, db(psd[f"cam{i}"]), lw=1.3, label=f"cam{i}")
    a2.set_title("8 mikrofonů kamer"); a2.set_xlim(30, f.max()); a2.set_xlabel("Hz")
    a2.legend(ncol=2, fontsize=7); a2.grid(alpha=0.2)
    fig.tight_layout(); fig_to_st(fig)

    st.header("Jak dobře každý kanál čte úroveň ucpání")
    st.caption("Po kanálech, křížová validace seskupená podle konfigurace (každý kanál "
               "zvlášť). Vyšší = tento senzor čte úroveň lépe. Rozptyl mezi mikrofony "
               "je velký; mikrofony kamer jsou jednotně silné.")
    rows = []
    for d, v in ch.items():
        rows.append({"kanál": d, "rodina": "kamera" if d.startswith("cam") else "mikrofon",
                     "hladina (dB)": round(v["features"]["rms_db"], 1),
                     "tón čerpadla (Hz)": round(v["features"]["tone_freq"]),
                     "výtlak ρ": round(v["discharge"]["spearman"], 2) if v["discharge"] else None,
                     "výtlak do ±1": round(v["discharge"]["within1"], 2) if v["discharge"] else None,
                     "sání ρ": round(v["suction"]["spearman"], 2) if v["suction"] else None})
    tab = pd.DataFrame(rows)
    fig2, ax = plt.subplots(figsize=(11, 3.6))
    order = tab.sort_values(["rodina", "výtlak ρ"], ascending=[True, False])
    colors = ["#7fd1ff" if fam == "kamera" else "#f0b54b" for fam in order["rodina"]]
    ax.bar(order["kanál"], order["výtlak ρ"], color=colors)
    for i, val in enumerate(order["výtlak ρ"]):
        ax.text(i, val + 0.01, f"{val:.2f}", ha="center", fontsize=7)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Spearman ρ úrovně výtlaku (po kanálech)")
    ax.set_title("Kvalita čtení ucpání po kanálech  (oranžová = mikrofon, modrá = kamera)")
    ax.grid(axis="y", alpha=0.3); fig2.tight_layout(); fig_to_st(fig2)
    st.dataframe(order, hide_index=True, width="stretch")

    bm = tab[tab.rodina == "mikrofon"].sort_values("výtlak ρ")
    bc = tab[tab.rodina == "kamera"].sort_values("výtlak ρ")
    st.markdown(f"""
**Závěry**
- Mezi 8 mikrofony se kvalita čtení výtlaku pohybuje od **{bm.iloc[0]['kanál']}
  (ρ={bm.iloc[0]['výtlak ρ']})** po **{bm.iloc[-1]['kanál']} (ρ={bm.iloc[-1]['výtlak ρ']})** —
  samotné umístění to mění zhruba 2×.
- Mikrofony kamer jsou jednotně silné (nejlepší **{bc.iloc[-1]['kanál']}
  ρ={bc.iloc[-1]['výtlak ρ']}**), jsou ~30 dB hlasitější a tón zátěže čerpadla mají
  kolem ~800 Hz — čtou výtlak nejlépe.
- Proto se modely trénují **zvlášť pro každý typ senzoru**, proto míchání všech 16
  kanálů zhoršilo přesné čtení úrovně a proto pomáhá fúze kanálů (nebo nasazení
  nejlépe umístěného). Živý posluchač automaticky detekuje *typ*; konkrétní umístění
  je rozhodnutí při nasazení.
""")


def page_models():
    st.title("🧠 Modely a jak jsme je trénovali")
    st.markdown("""
### Příznaky — 30 deskriptorů nezávislých na hlasitosti na nahrávku
Z jednoho kanálu extrahujeme: 12 logaritmických energií v pásmech (0→22 kHz, jemněji
pod 350 Hz), spektrální těžiště/šířku/rolloff/plochost/crest, ZCR, frekvenci a
výraznost dominantního tónu čerpadla + poměry 2./3. harmonické, hrubé poměry pásem a
energii amplitudové modulace obálky. **Všechny příznaky kromě jednoho jsou relativní
k celkovému výkonu nebo jsou to poměry**, takže se modely neopírají o zisk kanálu
(vynechání jediného hlasitostního příznaku přesnost nezmění).

### Model — gradientně boostované stromy, zvlášť pro každý typ senzoru
`HistGradientBoosting` (regrese pro úroveň, klasifikace pro přítomnost). Mikrofony a
mikrofony kamer se trénují **odděleně** — umístění rozhoduje o kvalitě (kamery čtou
výtlak nejlépe). Drobný **automatický detektor mic vs kamera** (98 % na okno) umožní
živému posluchači vybrat si správný typ sám.

### Dvě analytická okna
| úloha | okno | proč |
|---|---|---|
| aerace zap/vyp, který stroj | **8 s** | rychlé; trénované na 8s oknech |
| **úroveň** ucpání | **až 60 s** | potřebuje dlouhé okno — 8 s ztrácí příliš mnoho, hlavně u mikrofonů (do ±1 při oddělení dne 0,6 → **1,0** při 60 s) |

### Validace odolná vůči úniku dat (tři úrovně přísnosti)
- **seskupení podle konfigurace** — oddělíme celou konfiguraci ventilů (+ její
  zašuměné „dvojníky“).
- **oddělení celé kampaně** — trénink na některých nahrávacích *dnech*, test na jiném dni.
- **oddělení celé úrovně** — oddělíme celou úroveň omezení (test interpolace).

Zásadní je, že každá nahrávka spustí 16 senzorů v jednom okamžiku, takže **všechny
skupiny v CV drží tyto sourozence pohromadě** — naivní rozdělení přesnost nadhodnotí
(0,977 → 0,993).

### Trénink s „těžkými negativy“ (aerace / stroje)
První verze, trénované jen v čisté kampani 5_25, dávaly ~19 % falešných poplachů na
mikrofonu, když byl M1 přiškrcený. Přidání nahrávek pouze s M1 (které jsou jistě bez
aerace / bez pomocných strojů) jako negativů snížilo **falešné poplachy na ~0 %** při
zachování citlivosti.
""")
    st.caption("Balíčky modelů jsou uloženy podle úlohy a senzoru v run/models/ — "
               "`{scaler, model, features, levels}`; lze přímo načíst a predikovat.")


def page_results():
    st.title("📊 Výsledky")
    res = R.load_json("results.json")
    live = R.load_json("live_test_results.json")

    st.header("Úroveň ucpání (hlavní výsledek)")
    rows = []
    sc = res.get("severity_leave_campaign_out", {})
    for axis, cz in [("discharge", "výtlak"), ("suction", "sání")]:
        for dv in ["mic", "cam"]:
            m = sc.get(axis, {}).get(dv, {}).get("pooled", {})
            if m:
                rows.append({"úloha": f"úroveň {cz}", "senzor": dv,
                             "do ±1": round(m["within1"], 3),
                             "přesně": round(m["exact_acc"], 3),
                             "MAE (kroky)": round(m["MAE_steps"], 2),
                             "test": "oddělení kampaně"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    for col, fn, cap in [(c1, "fig_severity_discharge.png", "Výtlak: predikce vs skutečnost"),
                         (c2, "fig_severity_suction.png", "Sání: predikce vs skutečnost")]:
        p = os.path.join(RUN, fn)
        if os.path.exists(p):
            col.image(p, caption=cap, width="stretch")
    st.markdown("**Odolnost vůči hluku** (trénink na čistých, test na každém hluku "
                "z prostředí) a kontrola interpolace s oddělením celé úrovně:")
    c3, c4 = st.columns(2)
    for col, fn in [(c3, "fig_robustness.png"), (c4, "fig_leave_level_out.png")]:
        p = os.path.join(RUN, fn)
        if os.path.exists(p):
            col.image(p, width="stretch")

    st.header("Aerace zap/vyp a který stroj")
    c5, c6 = st.columns(2)
    for col, fn, cap in [(c5, "fig_aeration.png", "Aerace: shodná konfigurace, nezávislé na hlasitosti"),
                         (c6, "fig_which_pump.png", "Který stroj: F1 pro každý stroj")]:
        p = os.path.join(RUN, fn)
        if os.path.exists(p):
            col.image(p, caption=cap, width="stretch")

    if live:
        st.subheader("Posluchač od začátku do konce (oddělené nahrávky)")
        rows = []
        a = live.get("aeration_matched_gain_invariant", {})
        for dv in ["mic", "cam"]:
            if dv in a:
                rows.append({"úloha": "aerace zap/vyp", "senzor": dv,
                             "přesnost 1 mic": round(a[dv]["clip_acc_single_sensor"], 3),
                             "přesnost fúze 8 kanálů": round(a[dv]["clip_acc_fused_8ch"], 3)})
        mp = live.get("machine_presence", {})
        for mm in ["M2", "M3", "M4"]:
            for dv in ["mic", "cam"]:
                if mm in mp and dv in mp[mm]:
                    rows.append({"úloha": f"{mm} běží", "senzor": dv,
                                 "přesnost 1 mic": round(mp[mm][dv]["clip_acc_single_sensor"], 3),
                                 "přesnost fúze 8 kanálů": round(mp[mm][dv]["clip_acc_fused_8ch"], 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    p = os.path.join(RUN, "fig_importance.png")
    if os.path.exists(p):
        st.subheader("Co řídí jednotlivá čtení (důležitost příznaků)")
        st.image(p, width="stretch")


def page_demo():
    st.title("🎧 Živá ukázka — modely na reálných nahrávkách")
    st.caption("Spustí natrénované modely **přímo na nahrávce z datasetu** (spolehlivá, "
               "ověřená cesta) a porovná predikci se skutečností.")
    if not (R.audio_available() and R.models_available()):
        st.info("**Tato hostovaná verze neobsahuje živou ukázku nad soubory.** "
                "Potřebuje surová data (~63 GB WAV) a balíčky modelů, které nejsou "
                "v repozitáři. **Přesnost** modelů je na stránce *Výsledky* a stránky "
                "*Akustické signatury / Aerace / Ucpání* ukazují přesně, na co se modely "
                "dívají. Pro živou ukázku naklonujte repozitář s daty a spusťte lokálně "
                "`./run/run_report.sh`.")
        return
    lib = R.get_library(); lis = R.get_listener()

    c = st.columns(5)
    dev = c[0].selectbox("Senzor", ["cam", "mic"])
    vout = c[1].selectbox("Výtlak", ["libovolný"] + lib.options("valveOut"))
    vin = c[2].selectbox("Sání", ["libovolné"] + lib.options("valveIn"))
    aer = c[3].selectbox("Aerace", ["libovolná", "vyp", "zap"])
    noise = c[4].selectbox("Hluk", ["libovolný"] + list(NOISE_CZ))
    cc = st.columns(3)
    m2 = cc[0].checkbox("M2 čerpadlo"); m3 = cc[1].checkbox("M3 ventilátor")
    m4 = cc[2].checkbox("M4 ventilátor")

    def sel(v):
        return None if v in ("libovolný", "libovolné", "libovolná") else v
    aerv = None if aer == "libovolná" else (1 if aer == "zap" else 0)

    if "demo_clip" not in st.session_state:
        st.session_state.demo_clip = None
    if st.button("🔎 Najít odpovídající nahrávku a spustit modely", type="primary"):
        st.session_state.demo_clip = lib.find(
            dev_type=dev, valveOut=sel(vout), valveIn=sel(vin), aeration=aerv,
            noise_cat=sel(noise), M2=int(m2), M3=int(m3), M4=int(m4))

    clip = st.session_state.demo_clip
    if clip is None:
        st.info("Zvolte podmínku a stiskněte tlačítko. Zkuste: Senzor=cam, Aerace=zap "
                "(příklad, který selže přes smyčku zvuku, ale tady funguje), nebo "
                "Výtlak=11 pro silně ucpanou nahrávku.")
        return

    import soundfile as sf
    x, _ = sf.read(clip["path"], dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    r = lis.analyze(x[:int(60 * SR)])

    st.markdown(f"**Nahrávka:** `{clip['file']}`")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### Skutečnost (ground truth)")
        st.markdown(f"""
- sání (valveIn) = **L{clip['valveIn']}**
- výtlak (valveOut) = **L{clip['valveOut']}**
- aerace = **{'ZAP' if clip['aeration'] else 'vyp'}**
- běží = **M1{''.join(f' + {m}' for m in ['M2', 'M3', 'M4'] if clip[m])}**
- senzor = **{clip['dev_type']}**, hluk = **{clip['noise']}**""")
    with g2:
        st.markdown("##### Predikce modelu (ze zvuku)")
        b = r.get("blockage", {}); aerp = r["aeration"]
        run = [m for m in r["running"] if m != "M1"]
        unc = b.get("discharge", {}).get("reliability") == "uncertain_aux"
        st.markdown(f"""
- automaticky detekovaný senzor = **{r['sensor']}** {'✅' if r['sensor'] == clip['dev_type'] else '⚠️'}
- aerace = **{'ZAP' if aerp['on'] else 'vyp'}** (p={aerp['p']:.2f}) {'✅' if aerp['on'] == bool(clip['aeration']) else '⚠️'}
- běží = **M1{''.join(f' + {m}' for m in run)}**
- úroveň výtlaku = **{('L' + str(b['discharge']['level'])) if b else '—'}** {'(nejisté · běží pomocný stroj)' if unc else ''}
- úroveň sání = **{('L' + str(b['suction']['level'])) if b else '—'}**""")

    n = min(len(x), 12 * SR)
    f, t, Sxx = ssignal.spectrogram(x[:n], SR, nperseg=2048, noverlap=1024)
    keep = f <= 8000
    S = 10 * np.log10(Sxx[keep] + 1e-12)
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.imshow(S, origin="lower", aspect="auto", extent=[0, n / SR, 0, 8000],
              cmap="magma", vmin=S.max() - 75, vmax=S.max())
    ax.set_ylabel("Hz"); ax.set_xlabel("čas (s)"); ax.set_title("Spektrogram (prvních 12 s)")
    fig_to_st(fig)
    if clip["aeration"]:
        st.warning("Pozn.: modely ucpání jsou trénovány na zvuku pouze s M1, takže když "
                   "běží aerace (nebo pomocný stroj), je čtení ucpání označeno jako "
                   "nejisté, nikoliv důvěryhodné.")


def page_aeration_compare():
    import soundfile as sf
    st.title("💨 Aerace: jak ji poznat")
    st.markdown("""
Přímé vizuální A/B: **stejné čerpadlo při stejném nastavení ventilů**, nahrané se
vzduchovým injektorem (aerací) **VYP vs ZAP**. Sledujte nízké frekvence.""")
    dt = st.radio("Senzor", ["cam", "mic"], horizontal=True)
    have = R.audio_available()
    xo = xn = None; cap = ""
    if have:
        if st.button("🎲 Vybrat jiný pár zap/vyp"):
            st.session_state.pop("aer_pair_cz", None)
        files = R.aeration_files(dt)
        if not files["on"] or not files["off"]:
            have = False
        else:
            if "aer_pair_cz" not in st.session_state or st.session_state.get("aer_dt_cz") != dt:
                import random
                st.session_state.aer_pair_cz = (random.choice(files["off"]),
                                                random.choice(files["on"]))
                st.session_state.aer_dt_cz = dt
            off_p, on_p = st.session_state.aer_pair_cz

            def load(p):
                y, sr = sf.read(p, dtype="float32")
                return y.mean(axis=1) if y.ndim > 1 else y
            xo, xn = load(off_p), load(on_p)
            fo, po = ssignal.welch(xo - xo.mean(), SR, nperseg=8192, noverlap=4096)
            fn, pn = ssignal.welch(xn - xn.mean(), SR, nperseg=8192, noverlap=4096)
            cap = f"VYP: `{os.path.basename(off_p)}`  ·  ZAP: `{os.path.basename(on_p)}`"
    if not have:
        sigs = R.load_sigs()
        fo = fn = sigs["freq"]; po = sigs[f"{dt}_aer_off"]; pn = sigs[f"{dt}_aer_on"]
        cap = "_hostovaná verze: průměrná spektra přes shodné nahrávky — pro zobrazení " \
              "po souborech a spektrogramy spusťte lokálně_"

    st.subheader("Spektrum — aerace VYP vs ZAP")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.4))
    a1.semilogx(fo, db(po), color="#7fd1ff", lw=1.6, label="aerace VYP")
    a1.semilogx(fn, db(pn), color="#46d17a", lw=1.6, label="aerace ZAP")
    a1.axvspan(50, 100, color="#f0b54b", alpha=0.15)
    a1.set_xlim(30, 12000); a1.set_xlabel("Hz"); a1.set_ylabel("výkon (dB, rel.)")
    a1.legend(); a1.grid(alpha=0.2); a1.set_title("celé spektrum (pásmo 50–100 Hz zvýrazněno)")
    a2.semilogx(fn, db(pn) - db(po), color="#f0b54b", lw=1.6)
    a2.axhline(0, color="#777", lw=0.7); a2.axvspan(50, 100, color="#f0b54b", alpha=0.15)
    a2.set_xlim(30, 12000); a2.set_xlabel("Hz"); a2.set_ylabel("Δ výkon (dB)")
    a2.set_title("Δ = ZAP − VYP"); a2.grid(alpha=0.2)
    fig.tight_layout(); fig_to_st(fig)

    if xo is not None:
        st.subheader("Spektrogram — nízké pásmo (0–1500 Hz), kde se to projeví")
        fig2, (b1, b2) = plt.subplots(1, 2, figsize=(12, 3.0))
        R._spec(b1, xo, 1500, "aerace VYP"); R._spec(b2, xn, 1500, "aerace ZAP")
        fig2.tight_layout(); fig_to_st(fig2)

    def band_db(f, p, lo, hi):
        return 10 * np.log10(p[(f >= lo) & (f < hi)].sum() / p.sum() + 1e-12)
    d = band_db(fn, pn, 50, 100) - band_db(fo, po, 50, 100)
    st.caption(f"{cap}  · změna pásma 50–100 Hz: **{d:+.1f} dB**")

    st.subheader("Co v signálu prozradí, že aerace běží")
    st.markdown("""
Z analýzy shodných párů zap/vyp napříč oběma typy senzorů se při zapnutí **vzduchového
injektoru** současně mění tři věci — a fyzikálně přesně to, co byste čekali od vhánění
vzduchu do vody:

1. **Zesílení v pásmu 50–100 Hz (≈ +4,5 dB).** Nejkonzistentnější znak u *obou* typů
   senzorů — nízkofrekvenční bublání/dunění, které při vypnuté aeraci prostě není.
   *(Zvýrazněné pásmo v grafech výše.)*
2. **Dominantní spektrální vrchol klesne dolů.** Hlavní tón čerpadla spadne z ~800 Hz
   na ~85 Hz u kamer (a ~250 → ~150 Hz u mikrofonů): vháněný vzduch čerpadlo „přitíží“,
   takže jeho nejhlasitější rezonance se posune do bublajícího pásma. Ve spektrogramu
   vidíte, jak se energie **posune dolů**.
3. **Zvuk je méně „tónový“, více bublavý.** Spektrální crest klesne ~5 dB (jediný ostrý
   vrchol se zaplní) a — nejjasněji u kvalitnějších mikrofonů kamer — obálka získá
   silnou **amplitudovou modulaci 2–40 Hz (≈ +8 dB)**: rytmické bublání bublin.

Jednou větou: **aerace zap = rozsvítí se nízké pásmo (≈50–100 Hz) a dominantní tón se
přesune tam dolů, s slyšitelnou několikahertzovou modulací bublání.** Protože jde
o změny *tvaru/modulace* (ne hlasitosti), je detektor nezávislý na zisku a funguje,
i když se absolutní hlasitost téměř nemění.

*Upozornění: aerace byla nahrána jen při jednom nastavení ventilů (`vin1/vout1`); když
je M1 silně přiškrcený, základní spektrum se už samo posouvá dolů, proto živý posluchač
označí ucpání jako nejisté, kdykoliv slyší aeraci.*
""")


def page_blockage_compare():
    import soundfile as sf
    st.title("🚰 Ucpání: jak ho poznat")
    st.markdown("""
M1 má dva škrticí ventily — jeden na **sací** (vstupní, `valveIn`) straně a jeden na
**výtlačné** (výstupní, `valveOut`) straně. Oba omezují průtok, ale dělají se zvukem
*různé* věci. Zde je, jak se každý vyvíjí při zavírání (1 = otevřeno → více omezeno) a
jak se **sání-při-x liší od výtlaku-při-x**.""")
    sigs = R.load_sigs()
    dt = st.radio("Senzor", ["cam", "mic"], horizontal=True, key="blk_dt_cz")
    f = sigs["freq"]

    st.header("1 · Jak se spektrum vyvíjí při zavírání ventilu (x++)")
    c1, c2 = st.columns(2)
    with c1:
        fig_to_st(R._psd_sweep(sigs, dt, "discharge", [1, 2, 3, 4, 5, 8, 11],
                               "VÝTLAK (valveOut) — zavírání výstupu", "vout"))
        st.caption("Protitlak na výstupu nutí čerpadlo „dřít“ → **střední pásmo "
                   "250–500 Hz roste** a tón se posouvá. Velké, monotónní, snadno čitelné.")
    with c2:
        fig_to_st(R._psd_sweep(sigs, dt, "suction", [1, 2, 3, 4, 5],
                               "SÁNÍ (valveIn) — hladovění vstupu", "vin"))
        st.caption("Hladovění vstupu → **dominantní tón klesá na velmi nízkou frekvenci**"
                   + (" a roste **šum kavitace 4–8 kHz**" if dt == "mic" else "") +
                   ". Jemnější — těžší osa.")

    st.header("2 · Trendy, kvantitativně")

    def bdb(p, lo, hi):
        return 10 * np.log10(p[(f >= lo) & (f < hi)].sum() / p.sum() + 1e-12)

    def tone(p):
        lm = f < 2000
        return f[lm][np.argmax(p[lm])]
    dlv = [1, 2, 3, 4, 5, 8, 11]; slv = [1, 2, 3, 4, 5]
    series = [
        ([bdb(sigs[f"{dt}_discharge_{l}"], 250, 500) for l in dlv],
         [bdb(sigs[f"{dt}_suction_{l}"], 250, 500) for l in slv],
         "Střední pásmo 250–500 Hz", "výkon (dB, rel.)"),
        ([tone(sigs[f"{dt}_discharge_{l}"]) for l in dlv],
         [tone(sigs[f"{dt}_suction_{l}"]) for l in slv], "Dominantní tón čerpadla", "Hz"),
        ([bdb(sigs[f"{dt}_discharge_{l}"], 4000, 8000) for l in dlv],
         [bdb(sigs[f"{dt}_suction_{l}"], 4000, 8000) for l in slv],
         "Vysoké pásmo 4–8 kHz", "výkon (dB, rel.)")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.0))
    for ax, (yd, ys, ttl, yl) in zip(axs, series):
        ax.plot(dlv, yd, "o-", color="#ff9d6b", label="výtlak")
        ax.plot(slv, ys, "s-", color="#7fd1ff", label="sání")
        ax.set_title(ttl); ax.set_xlabel("úroveň ventilu (1=otevřeno)"); ax.set_ylabel(yl)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)
    fig.tight_layout(); fig_to_st(fig)
    st.caption("Výtlak ⟶ střední pásmo roste monotónně (hlavní signál „závažnosti“). "
               "Sání ⟶ tón prudce klesá" +
               (" a roste šum 4–8 kHz" if dt == "mic" else "") +
               " — odlišné otisky pro stejný úkon „zavření ventilu“.")

    st.header("3 · Sání-při-x vs výtlak-při-x — liší se? (ano)")
    x = st.select_slider("Porovnat při úrovni omezení x =", options=[2, 3, 4, 5], value=4)
    ps = sigs[f"{dt}_suction_{x}"]; pd_ = sigs[f"{dt}_discharge_{x}"]
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.3))
    a1.semilogx(f, db(ps), color="#7fd1ff", lw=1.7, label=f"SÁNÍ při {x} (vin={x},vout=1)")
    a1.semilogx(f, db(pd_), color="#ff9d6b", lw=1.7, label=f"VÝTLAK při {x} (vin=1,vout={x})")
    a1.axvspan(250, 500, color="#ff9d6b", alpha=0.10)
    a1.set_xlim(30, 12000); a1.set_xlabel("Hz"); a1.set_ylabel("výkon (dB, rel.)")
    a1.legend(fontsize=8); a1.grid(alpha=0.2); a1.set_title(f"stejná nominální úroveň x={x}")
    a2.semilogx(f, db(pd_) - db(ps), color="#46d17a", lw=1.6); a2.axhline(0, color="#777", lw=0.7)
    a2.set_xlim(30, 12000); a2.set_xlabel("Hz"); a2.set_ylabel("Δ výkon (dB)")
    a2.set_title("Δ = výtlak − sání"); a2.grid(alpha=0.2)
    fig2.tight_layout(); fig_to_st(fig2)
    st.caption(f"Při stejném x={x}: výtlak nese **více středního pásma 250–500 Hz** "
               f"(tón ≈{tone(pd_):.0f} Hz), zatímco sání je **níže posazené** "
               f"(tón ≈{tone(ps):.0f} Hz). Nejsou to stejné zvuky — proto obě osy "
               "čtou oddělené modely.")

    if R.audio_available():
        st.subheader("Reálné nahrávky — spektrogramy (0–1500 Hz)")
        sfp = R.blockage_file(dt, x, 1); dfp = R.blockage_file(dt, 1, x)
        if sfp and dfp:
            def load(p):
                y, sr = sf.read(p, dtype="float32"); return y.mean(1) if y.ndim > 1 else y
            fig3, (b1, b2) = plt.subplots(1, 2, figsize=(12, 3.0))
            R._spec(b1, load(sfp), 1500, f"SÁNÍ při {x} (vin={x}, vout=1)")
            R._spec(b2, load(dfp), 1500, f"VÝTLAK při {x} (vin=1, vout={x})")
            fig3.tight_layout(); fig_to_st(fig3)

    st.header("Co signál prozradí")
    st.markdown("""
- **Výtlačné (výstupní) ucpání** = protitlak na výstupu. Při zavírání čerpadlo pracuje
  „proti zdi“, takže jeho **tónová energie ve středním pásmu 250–500 Hz roste se
  závažností** a tón se v extrémech posune nahoru. Je velké, monotónní a hlasité →
  model čte *přesnou* úroveň do ±1 prakticky vždy (kamera přesně ≈0,98).
- **Sací (vstupní) ucpání** = hladovění vstupu. Pracovní bod se posune tak, že
  **dominantní tón klesne na velmi nízkou frekvenci** (až ~80–120 Hz) a u mikrofonů
  roste **šum kavitace 4–8 kHz**, jak hladovějící vstup kavituje. Efekt je reálný, ale
  jemnější a zašuměnější → čitelný do ±1 vždy, ale přesný krok je těžší („těžká osa“).
- **Při stejném x se rozlišují:** výtlak = *střední pásmo + vyšší tón*, sání = *nízký
  tón (+ šum VF u mikrofonů)*. Vstupní a výstupní omezení tedy nejsou zaměnitelné —
  report je modeluje jako **dvě nezávislé ordinální osy**, ne jedno společné „ucpání“.
""")


PAGES = {
    "① Přehled — příběh": page_overview,
    "② Dataset a parametry": page_dataset,
    "③ Akustické signatury": page_signatures,
    "④ Rozdíly mezi mikrofony": page_channels,
    "⑤ Modely a trénink": page_models,
    "⑥ Výsledky": page_results,
    "⑦ Živá ukázka": page_demo,
    "⑧ Aerace: jak ji poznat": page_aeration_compare,
    "⑨ Ucpání: jak ho poznat": page_blockage_compare,
}


def main():
    st.sidebar.title("Pool-Audio")
    st.sidebar.caption("Čtení provozního stavu čerpadla ze zvuku")
    choice = st.sidebar.radio("Sekce reportu", list(PAGES))
    st.sidebar.divider()
    st.sidebar.caption("Modely v run/models/ · data: 13 850 souborů, 7 kampaní · "
                       "veškeré zařízení je zdravé (provozní konfigurace, ne poruchy).")
    PAGES[choice]()


if __name__ == "__main__":
    main()
