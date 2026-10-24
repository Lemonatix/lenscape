"""lenscape -- interaktiver Gravitationslinsen-Simulator.

Start:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from lenscape.catalogs import DATA_DIR, load_slacs
from lenscape.caustics import caustics, critical_curves
from lenscape.cosmology import PLANCK18_H0, PLANCK18_OM0, LensGeometry, make_cosmology
from lenscape.lenses import NFW, SIE, SIS, ExternalShear, PointMass, einstein_radius
from lenscape.raytrace import convergence_map, magnification_map, make_grid, render, render_unlensed
from lenscape.solver import find_images
from lenscape.sources import SersicSource
from lenscape.strong_lenses import einstein_quantities

st.set_page_config(page_title="lenscape", page_icon="🔭", layout="wide")

plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "black",
    "axes.edgecolor": "#888",
    "axes.labelcolor": "#ddd",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "text.color": "#ddd",
    "font.size": 9,
})

# ---------------------------------------------------------------- Sidebar
sb = st.sidebar
sb.title("🔭 lenscape")

sb.header("Kosmologie")
H0 = sb.slider("H₀ [km/s/Mpc]", 50.0, 90.0, PLANCK18_H0, 0.1)
Om0 = sb.slider("Ωₘ (Materie inkl. Dunkler Materie)", 0.05, 1.0, PLANCK18_OM0, 0.01)
flat = sb.checkbox("flaches Universum (Ω_Λ = 1 − Ωₘ)", value=True)
Ode0 = None if flat else sb.slider("Ω_Λ (Dunkle Energie)", 0.0, 1.2, 1 - PLANCK18_OM0, 0.01)
w0 = sb.slider("w (Zustandsgleichung Dunkle Energie)", -2.0, -0.33, -1.0, 0.01)
cosmo = make_cosmology(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0)

HAVE_SLACS = (DATA_DIR / "slacs_bolton2008" / "table5.dat").exists()


@st.cache_data(show_spinner=False)
def slacs_table():
    return load_slacs().to_pandas()


@st.cache_data(show_spinner=False)
def slacs_physics(H0, Om0, Ode0, w0):
    from astropy.table import Table

    t = Table.from_pandas(slacs_table())
    return einstein_quantities(t, make_cosmology(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0)).to_pandas()


sb.header("Linse")
modes = ["Eigene Linse", "Echte Linse (SLACS)"] if HAVE_SLACS else ["Eigene Linse"]
mode = sb.radio("Linsenquelle", modes, horizontal=True, label_visibility="collapsed")
real = None

if mode.startswith("Echte"):
    cat = slacs_table()
    name = sb.selectbox(f"SLACS-Linse ({len(cat)} Grade-A-Systeme)", list(cat["name"]),
                        index=list(cat["name"]).index("J0946+1006") if "J0946+1006" in list(cat["name"]) else 0)
    real = cat[cat["name"] == name].iloc[0]
    z_d, z_s = float(real["z_d"]), float(real["z_s"])
    geom = LensGeometry(z_d, z_s, cosmo)
    model = "SIE (elliptische Galaxie)"
    # Positionswinkel: Grad östlich von Nord; im Bild ist Nord oben, Ost links
    main = SIE(float(real["theta_E"]), q=float(real["q"]), phi=np.radians(90 + float(real["pa"])))
    param_text = f"SDSS {name}: θ_E = {real['theta_E']}″, q = {real['q']}, PA = {real['pa']}° (Bolton et al. 2008)"
    sb.caption(f"z_d = {z_d}, z_s = {z_s}, σ_SDSS = {real['sigma']:.0f} km/s. "
               "θ_E ist beobachtet; die Kosmologie-Regler ändern daher die abgeleitete Masse, nicht das Bild.")
else:
    sb.subheader("Geometrie")
    z_d = sb.slider("Rotverschiebung Linse z_d", 0.05, 2.0, 0.3, 0.01)
    z_s = sb.slider("Rotverschiebung Quelle z_s", 0.1, 5.0, 1.0, 0.01)
    if z_s <= z_d + 0.01:
        st.error("Die Quelle muss hinter der Linse liegen (z_s > z_d).")
        st.stop()
    geom = LensGeometry(z_d, z_s, cosmo)

    sb.subheader("Massenmodell")
    model = sb.selectbox("Massenmodell", ["SIE (elliptische Galaxie)", "SIS (Galaxie)", "Punktmasse", "NFW (Dunkle-Materie-Halo)"])
    if model.startswith("Punktmasse"):
        logM = sb.slider("log₁₀ M [M☉]", 6.0, 14.0, 11.5, 0.05)
        main = PointMass.from_mass(geom, 10**logM)
        param_text = f"M = 10^{logM:.2f} M☉"
    elif model.startswith("SIS"):
        sv = sb.slider("Geschwindigkeitsdispersion σ_v [km/s]", 50, 1500, 250, 5)
        main = SIS.from_sigma_v(geom, sv)
        param_text = f"σ_v = {sv} km/s"
    elif model.startswith("SIE"):
        sv = sb.slider("Geschwindigkeitsdispersion σ_v [km/s]", 50, 1500, 250, 5)
        q = sb.slider("Achsenverhältnis q", 0.2, 1.0, 0.75, 0.01)
        phi = np.radians(sb.slider("Orientierung [°]", 0, 180, 30))
        main = SIE(geom.theta_E_sis(sv), q=q, phi=phi)
        param_text = f"σ_v = {sv} km/s, q = {q:.2f}"
    else:
        logM = sb.slider("log₁₀ M₂₀₀ [M☉]", 11.0, 15.8, 14.8, 0.05)
        conc = sb.slider("Konzentration c", 1.0, 20.0, 6.0, 0.1)
        main = NFW.from_m200(geom, 10**logM, conc)
        param_text = f"M₂₀₀ = 10^{logM:.2f} M☉, c = {conc:.1f}"

gamma = sb.slider("externe Scherung γ", 0.0, 0.3, 0.0, 0.005)
gphi = np.radians(sb.slider("Scherungswinkel [°]", 0, 180, 0)) if gamma > 0 else 0.0
lens = main + ExternalShear.from_polar(gamma, gphi) if gamma > 0 else main

theta_E = einstein_radius(main) if not isinstance(main, SIE) else main.theta_E
scale = theta_E if np.isfinite(theta_E) and theta_E > 1e-3 else 1.0

sb.header("Quelle (in Einheiten von θ_E)")
bx = sb.slider("Position x", -2.0, 2.0, 0.06, 0.01) * scale
by = sb.slider("Position y", -2.0, 2.0, 0.04, 0.01) * scale
R_e = sb.slider("Effektivradius R_e", 0.01, 1.0, 0.12, 0.01) * scale
n_s = sb.slider("Sérsic-Index n", 0.5, 4.0, 1.0, 0.1)
sq = sb.slider("Achsenverhältnis Quelle", 0.2, 1.0, 0.7, 0.01)
sphi = np.radians(sb.slider("Orientierung Quelle [°]", 0, 180, 60))
source = SersicSource(bx, by, R_e=R_e, n=n_s, q=sq, phi=sphi)

sb.header("Darstellung")
fov = sb.slider("Bildfeld (± in θ_E)", 1.0, 5.0, 2.2, 0.1) * scale
npix = sb.select_slider("Auflösung", [150, 200, 300, 400, 500], value=300)

# ---------------------------------------------------------------- Kopfzahlen
st.title("Gravitationslinsen-Simulator")
st.caption(f"{model} · {param_text} · z_d = {z_d}, z_s = {z_s} · H₀ = {H0}, Ωₘ = {Om0}, w = {w0}")

def sci(v: float, unit: str = "") -> str:
    """1.23e+11 -> 1.23·10¹¹"""
    exp = int(np.floor(np.log10(abs(v)))) if v else 0
    sup = str(exp).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    txt = f"{v:.4g}" if -2 <= exp <= 4 else f"{v / 10**exp:.2f}·10{sup}"
    return f"{txt} {unit}".strip()


c1, c2, c3, c4 = st.columns(4)
if np.isfinite(theta_E):
    c1.metric("Einsteinradius θ_E", f"{theta_E:.3g}″")
    c2.metric("θ_E physikalisch", sci(theta_E * geom.kpc_per_arcsec, "kpc"))
    c3.metric("Masse M(<θ_E)", sci(geom.mass_in_theta_E(theta_E), "M☉"))
else:
    c1.metric("Einsteinradius θ_E", "–", help="Linse ist subkritisch: kein Einsteinring")
c4.metric("Σ_crit [M☉/pc²]", sci(geom.sigma_crit.value))
st.caption(
    f"Winkeldurchmesser-Distanzen: D_d = {geom.D_d.value:.0f} Mpc · D_s = {geom.D_s.value:.0f} Mpc · "
    f"D_ds = {geom.D_ds.value:.0f} Mpc · Maßstab {geom.kpc_per_arcsec:.2f} kpc/″"
)

if real is not None:
    phys = slacs_physics(H0, Om0, Ode0, w0)
    row = phys[phys["name"] == real["name"]].iloc[0]
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("σ_SDSS (Sterne)", f"{row['sigma']:.0f} km/s" if np.isfinite(row["sigma"]) else "–")
    r2.metric("σ_SIE (Linse)", f"{row['sigma_SIE']:.0f} km/s",
              help="Geschwindigkeitsdispersion, die eine isotherme Linse mit diesem θ_E bräuchte")
    if "fDM_chab" in row and np.isfinite(row["fDM_chab"]):
        r3.metric("Dunkle Materie in R_E (Chabrier)", f"{100 * row['fDM_chab']:.0f} %")
        r4.metric("… (Salpeter-IMF)", f"{100 * row['fDM_salp']:.0f} %")


# ---------------------------------------------------------------- Kosmologie-Kurven (gecacht)
@st.cache_data(show_spinner=False)
def mass_vs_w(H0, Om0, Ode0, z_d, z_s, theta, n=25):
    ws = np.linspace(-2.0, -0.4, n)
    m = [LensGeometry(z_d, z_s, make_cosmology(H0, Om0, Ode0, w)).mass_in_theta_E(theta) for w in ws]
    return ws, np.array(m)


@st.cache_data(show_spinner=False)
def mass_vs_om(H0, Ode0, w0, z_d, z_s, theta, n=25):
    oms = np.linspace(0.1, 0.6, n)
    m = [LensGeometry(z_d, z_s, make_cosmology(H0, om, Ode0, w0)).mass_in_theta_E(theta) for om in oms]
    return oms, np.array(m)


@st.cache_data(show_spinner=False)
def efficiency_vs_zs(H0, Om0, Ode0, w0, z_d, n=40):
    cosmo = make_cosmology(H0, Om0, Ode0, w0)
    zs = np.linspace(z_d + 0.02, 5, n)
    return zs, np.array([(lambda g: (g.D_ds / g.D_s).value)(LensGeometry(z_d, z, cosmo)) for z in zs])


# ---------------------------------------------------------------- Plots
def draw_curves(ax, curves, **kw):
    for c in curves:
        ax.plot(c[:, 0], c[:, 1], lw=0.9, **kw)


tab_names = ["Simulator", "Konvergenz & Vergrößerung", "Dunkle Energie & Kosmologie"]
if HAVE_SLACS:
    tab_names.append("SLACS-Stichprobe")
tabs = st.tabs(tab_names)
tab_sim, tab_maps, tab_de = tabs[:3]

with tab_sim:
    crit = critical_curves(lens, fov, 400)
    caus = caustics(lens, curves=crit)
    img = render(lens, source, fov, npix, supersample=2)
    src_img = render_unlensed(source, fov, npix)
    images = find_images(lens, bx, by, half_width=fov, n=300)

    ext = [-fov, fov, -fov, fov]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.3))
    a1.imshow(src_img, origin="lower", extent=ext, cmap="inferno")
    draw_curves(a1, caus, color="#4fc3f7")
    a1.plot(bx, by, "+", color="white", ms=8)
    a1.set_title("Quellebene (ohne Linse) + Kaustiken")
    a2.imshow(img, origin="lower", extent=ext, cmap="inferno")
    draw_curves(a2, crit, color="#4fc3f7", ls="--")
    if len(images):
        a2.scatter(images[:, 0], images[:, 1], s=40, facecolors="none", edgecolors="#7CFC00")
    a2.plot(0, 0, "x", color="#bbb", ms=6)
    a2.set_title("Bildebene (beobachtet) + kritische Kurven")
    for a in (a1, a2):
        a.set_xlabel("x [arcsec]")
        a.set_ylabel("y [arcsec]")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if len(images):
        st.markdown(f"**{len(images)} Bild(er)** der Quellmitte (grüne Kreise):")
        st.dataframe(
            {
                "x [″]": np.round(images[:, 0], 4),
                "y [″]": np.round(images[:, 1], 4),
                "Vergrößerung μ": np.round(images[:, 2], 3),
                "Parität": np.where(images[:, 2] > 0, "+", "− (gespiegelt)"),
            },
            hide_index=True,
        )
    st.caption("Kreuzt die Quelle eine Kaustik (blau, links), ändert sich die Bildanzahl um zwei; "
               "auf den kritischen Kurven (gestrichelt, rechts) entstehen die hellen Bögen.")

with tab_maps:
    kap = convergence_map(lens, fov, npix)
    mu = magnification_map(lens, fov, npix)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.3))
    im1 = a1.imshow(np.log10(np.clip(kap, 1e-3, None)), origin="lower", extent=ext, cmap="viridis")
    a1.contour(*make_grid(fov, npix), kap, levels=[1.0], colors="white", linewidths=0.8)
    a1.set_title("log₁₀ κ  (weiß: κ = 1)")
    fig.colorbar(im1, ax=a1, fraction=0.046)
    im2 = a2.imshow(np.log10(np.abs(mu)), origin="lower", extent=ext, cmap="magma", vmin=-1, vmax=2)
    draw_curves(a2, crit, color="#4fc3f7", ls="--")
    a2.set_title("log₁₀ |μ|")
    fig.colorbar(im2, ax=a2, fraction=0.046)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_de:
    st.markdown(
        "Die Kosmologie steckt nur in den Distanzen. Hier siehst du, wie stark sich "
        "**Einsteinradius** (bei fester Linsenphysik) bzw. die **aus einem beobachteten θ_E "
        "abgeleitete Masse** mit Ωₘ und w verschieben."
    )
    obs_theta = theta_E if np.isfinite(theta_E) else 1.0
    ws, m_w = mass_vs_w(H0, Om0, Ode0, z_d, z_s, obs_theta)
    oms, m_om = mass_vs_om(H0, None if flat else Ode0, w0, z_d, z_s, obs_theta)
    m_ref = geom.mass_in_theta_E(obs_theta)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ws, m_w / m_ref, color="#ffb74d")
    a1.axvline(w0, color="#888", ls=":")
    a1.set_xlabel("w")
    a1.set_ylabel("M(<θ_E) / M_aktuell")
    a1.set_title(f"abgeleitete Masse bei festem θ_E = {obs_theta:.3g}″")
    a2.plot(oms, m_om / m_ref, color="#4fc3f7")
    a2.axvline(Om0, color="#888", ls=":")
    a2.set_xlabel("Ωₘ")
    a2.set_title("… in Abhängigkeit von Ωₘ")
    for a in (a1, a2):
        a.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    zs, ratio = efficiency_vs_zs(H0, Om0, Ode0, w0, z_d)
    fig, a = plt.subplots(figsize=(11, 3))
    a.plot(zs, ratio, color="#ce93d8")
    a.axvline(z_s, color="#888", ls=":")
    a.set_xlabel("z_s")
    a.set_ylabel("D_ds / D_s")
    a.set_title("Linseneffizienz: θ_E ∝ D_ds/D_s (SIS) — Quellen bei verschiedenen z vermessen die Kosmologie")
    a.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.metric("Zeitverzögerungs-Distanz D_Δt", f"{geom.time_delay_distance().value:.0f} Mpc",
              help="D_Δt ∝ 1/H₀ -- Grundlage der H₀-Messung aus gelinsten Quasaren.")


if HAVE_SLACS:
    with tabs[3]:
        st.markdown(
            "**63 echte Linsengalaxien** aus dem Sloan Lens ACS Survey (Bolton et al. 2008; Sternmassen: "
            "Auger et al. 2009), ausgewertet in der Kosmologie aus der Seitenleiste. "
            "Aus dem beobachteten Einsteinradius folgt die Gesamtmasse M_E; zieht man die Sternmasse ab, "
            "bleibt die Dunkle Materie."
        )
        phys = slacs_physics(H0, Om0, Ode0, w0)
        good = phys["good_sigma"].to_numpy(bool)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Median M_E", sci(float(np.median(phys["M_E"])), "M☉"))
        k2.metric("Median R_E", f"{np.median(phys['R_E_kpc']):.2f} kpc")
        k3.metric("⟨σ_SDSS / σ_SIE⟩", f"{phys['f_SIE'][good].mean():.3f}",
                  help="≈ 1 heißt: Sterne und Linse sehen dieselbe (isotherme) Gesamtmasse")
        k4.metric("Median f_DM (Chabrier)", f"{100 * np.nanmedian(phys['fDM_chab']):.0f} %")

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        a1.scatter(phys["sigma"][good], phys["M_E"][good], s=14, color="#4fc3f7")
        a1.set_xscale("log")
        a1.set_yscale("log")
        a1.set_xlabel("σ_SDSS [km/s]")
        a1.set_ylabel("M_E [M☉]")
        a1.set_title("Masse im Einsteinradius vs. Sterngeschwindigkeiten")
        for imf, col, lab in [("chab", "#4fc3f7", "Chabrier"), ("salp", "#ffb74d", "Salpeter")]:
            if f"fDM_{imf}" in phys:
                a2.scatter(phys["R_E_over_R_e"], phys[f"fDM_{imf}"], s=14, color=col, label=lab)
        a2.axhline(0, color="#888", lw=0.8)
        a2.set_xlabel("R_E / R_e")
        a2.set_ylabel("f_DM(<R_E)")
        a2.set_title("Anteil Dunkler Materie")
        a2.legend()
        for a in (a1, a2):
            a.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Ein negativer Dunkle-Materie-Anteil heißt: Die Sternmasse wäre größer als die Gesamtmasse. "
                   "Die Salpeter-IMF ist für diese Galaxien dann zu schwer. So schränken Linsen die Sternentstehung ein.")

        cols = ["name", "z_d", "z_s", "theta_E", "sigma", "sigma_SIE", "R_E_kpc", "M_E", "ML_E", "fDM_chab", "fDM_salp"]
        show = phys[[c for c in cols if c in phys]].copy()
        show["M_E"] = show["M_E"] / 1e11
        st.dataframe(
            show.rename(columns={"theta_E": "θ_E [″]", "sigma": "σ_SDSS", "sigma_SIE": "σ_SIE",
                                 "R_E_kpc": "R_E [kpc]", "M_E": "M_E [10¹¹ M☉]", "ML_E": "M/L_V",
                                 "fDM_chab": "f_DM Chab", "fDM_salp": "f_DM Salp"}).round(3),
            hide_index=True, width="stretch",
        )
