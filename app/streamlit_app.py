"""lenscape -- interactive gravitational-lensing simulator.

Run:  streamlit run app/streamlit_app.py

Formulas in captions, labels and metrics are written in LaTeX and rendered by
Streamlit (KaTeX); plot labels use matplotlib's mathtext with Computer Modern.
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
from lenscape.timedelay import time_delays

st.set_page_config(page_title="lenscape", layout="wide")

plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "black",
    "axes.edgecolor": "#888",
    "axes.labelcolor": "#ddd",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "text.color": "#ddd",
    "font.size": 9,
    "mathtext.fontset": "cm",
})


# ---------------------------------------------------------------- LaTeX helpers
def tex(s: str) -> str:
    """Inline math for Streamlit Markdown (KaTeX)."""
    return f"${s}$"


def num(v: float, digits: int = 3) -> str:
    """Number in LaTeX: 1.23e+11 -> 1.23 \\times 10^{11}; signs as proper minus."""
    if v == 0 or not np.isfinite(v):
        return "0" if v == 0 else r"\text{--}"
    exp = int(np.floor(np.log10(abs(v))))
    if 2 <= exp <= 4:
        return f"{v:.0f}"
    if -2 <= exp < 2:
        return f"{v:.{digits}g}"
    return f"{v / 10**exp:.2f}" + r" \times 10^{" + str(exp) + "}"


MSUN = r"\,M_\odot"
KMS = r"\,\mathrm{km\,s^{-1}}"
PCT = r"\,\%"
ARCSEC = "''"

# ---------------------------------------------------------------- sidebar
sb = st.sidebar
sb.title("lenscape")

sb.header("Cosmology")
H0 = sb.slider(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", 50.0, 90.0, PLANCK18_H0, 0.1)
Om0 = sb.slider(r"$\Omega_\mathrm{m}$ (matter incl. dark matter)", 0.05, 1.0, PLANCK18_OM0, 0.01)
flat = sb.checkbox(r"flat universe ($\Omega_\Lambda = 1 - \Omega_\mathrm{m}$)", value=True)
Ode0 = None if flat else sb.slider(r"$\Omega_\Lambda$ (dark energy)", 0.0, 1.2, 1 - PLANCK18_OM0, 0.01)
w0 = sb.slider(r"$w$ (dark-energy equation of state)", -2.0, -0.33, -1.0, 0.01)
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


sb.header("Lens")
modes = ["Custom lens", "Real lens (SLACS)"] if HAVE_SLACS else ["Custom lens"]
mode = sb.radio("Lens source", modes, horizontal=True, label_visibility="collapsed")
real = None

if mode.startswith("Real"):
    cat = slacs_table()
    name = sb.selectbox(f"SLACS lens ({len(cat)} grade-A systems)", list(cat["name"]),
                        index=list(cat["name"]).index("J0946+1006") if "J0946+1006" in list(cat["name"]) else 0)
    real = cat[cat["name"] == name].iloc[0]
    z_d, z_s = float(real["z_d"]), float(real["z_s"])
    geom = LensGeometry(z_d, z_s, cosmo)
    model = "SIE (elliptical galaxy)"
    # position angle: degrees east of north; in the image north is up, east is left
    main = SIE(float(real["theta_E"]), q=float(real["q"]), phi=np.radians(90 + float(real["pa"])))
    th, qq, pa = real["theta_E"], real["q"], real["pa"]
    param_text = (f"SDSS {name}: " + tex(r"\theta_\mathrm{E} = " + f"{th}{ARCSEC}") + ", " + tex(f"q = {qq}")
                  + ", " + tex(r"\mathrm{PA} = " + f"{pa}" + r"^\circ") + " (Bolton et al. 2008)")
    sb.caption(tex(r"z_\mathrm{d} = " + f"{z_d}") + ", " + tex(r"z_\mathrm{s} = " + f"{z_s}") + ", "
               + tex(r"\sigma_\mathrm{SDSS} = " + f"{real['sigma']:.0f}") + " km/s. "
               r"$\theta_\mathrm{E}$ is observed, so the cosmology sliders change the inferred mass, not the image.")
else:
    sb.subheader("Geometry")
    z_d = sb.slider(r"lens redshift $z_\mathrm{d}$", 0.05, 2.0, 0.3, 0.01)
    z_s = sb.slider(r"source redshift $z_\mathrm{s}$", 0.1, 5.0, 1.0, 0.01)
    if z_s <= z_d + 0.01:
        st.error(r"The source must lie behind the lens ($z_\mathrm{s} > z_\mathrm{d}$).")
        st.stop()
    geom = LensGeometry(z_d, z_s, cosmo)

    sb.subheader("Mass model")
    model = sb.selectbox("Mass model", ["SIE (elliptical galaxy)", "SIS (galaxy)", "Point mass",
                                        "NFW (dark-matter halo)"])
    if model.startswith("Point"):
        logM = sb.slider(r"$\log_{10} M\ [M_\odot]$", 6.0, 14.0, 11.5, 0.05)
        main = PointMass.from_mass(geom, 10**logM)
        param_text = tex("M = 10^{" + f"{logM:.2f}" + "}" + MSUN)
    elif model.startswith("SIS"):
        sv = sb.slider(r"velocity dispersion $\sigma_v$ [km/s]", 50, 1500, 250, 5)
        main = SIS.from_sigma_v(geom, sv)
        param_text = tex(rf"\sigma_v = {sv}") + " km/s"
    elif model.startswith("SIE"):
        sv = sb.slider(r"velocity dispersion $\sigma_v$ [km/s]", 50, 1500, 250, 5)
        q = sb.slider(r"axis ratio $q$", 0.2, 1.0, 0.75, 0.01)
        phi = np.radians(sb.slider("orientation [°]", 0, 180, 30))
        main = SIE(geom.theta_E_sis(sv), q=q, phi=phi)
        param_text = tex(rf"\sigma_v = {sv}") + " km/s, " + tex(f"q = {q:.2f}")
    else:
        logM = sb.slider(r"$\log_{10} M_{200}\ [M_\odot]$", 11.0, 15.8, 14.8, 0.05)
        conc = sb.slider(r"concentration $c$", 1.0, 20.0, 6.0, 0.1)
        main = NFW.from_m200(geom, 10**logM, conc)
        param_text = tex("M_{200} = 10^{" + f"{logM:.2f}" + "}" + MSUN) + ", " + tex(f"c = {conc:.1f}")

gamma = sb.slider(r"external shear $\gamma$", 0.0, 0.3, 0.0, 0.005)
gphi = np.radians(sb.slider("shear angle [°]", 0, 180, 0)) if gamma > 0 else 0.0
lens = main + ExternalShear.from_polar(gamma, gphi) if gamma > 0 else main

theta_E = einstein_radius(main) if not isinstance(main, SIE) else main.theta_E
scale = theta_E if np.isfinite(theta_E) and theta_E > 1e-3 else 1.0

sb.header(r"Source (in units of $\theta_\mathrm{E}$)")
bx = sb.slider(r"position $\beta_x$", -2.0, 2.0, 0.06, 0.01) * scale
by = sb.slider(r"position $\beta_y$", -2.0, 2.0, 0.04, 0.01) * scale
R_e = sb.slider(r"effective radius $R_\mathrm{e}$", 0.01, 1.0, 0.12, 0.01) * scale
n_s = sb.slider(r"Sérsic index $n$", 0.5, 4.0, 1.0, 0.1)
sq = sb.slider(r"source axis ratio $q_\mathrm{s}$", 0.2, 1.0, 0.7, 0.01)
sphi = np.radians(sb.slider("source orientation [°]", 0, 180, 60))
source = SersicSource(bx, by, R_e=R_e, n=n_s, q=sq, phi=sphi)

sb.header("Display")
fov = sb.slider(r"field of view ($\pm$ in $\theta_\mathrm{E}$)", 1.0, 5.0, 2.2, 0.1) * scale
npix = sb.select_slider("resolution", [150, 200, 300, 400, 500], value=300)

# ---------------------------------------------------------------- headline numbers
st.title("Gravitational-lens simulator")
st.caption(f"{model} · {param_text} · "
           + tex(r"z_\mathrm{d} = " + f"{z_d:g}") + ", " + tex(r"z_\mathrm{s} = " + f"{z_s:g}") + " · "
           + tex(f"H_0 = {H0:.2f}") + ", " + tex(r"\Omega_\mathrm{m} = " + f"{Om0:.4g}") + ", "
           + tex(f"w = {w0:.2f}"))

c1, c2, c3, c4 = st.columns(4)
if np.isfinite(theta_E):
    c1.metric(r"Einstein radius $\theta_\mathrm{E}$", tex(f"{theta_E:.3g}{ARCSEC}"))
    c2.metric(r"$\theta_\mathrm{E}$ physical", tex(num(theta_E * geom.kpc_per_arcsec) + r"\,\mathrm{kpc}"))
    c3.metric(r"mass $M(<\theta_\mathrm{E})$ [$M_\odot$]", tex(num(geom.mass_in_theta_E(theta_E))))
else:
    c1.metric(r"Einstein radius $\theta_\mathrm{E}$", "–", help="The lens is subcritical: no Einstein ring")
c4.metric(r"critical density $\Sigma_\mathrm{crit}$ [$M_\odot\,\mathrm{pc}^{-2}$]", tex(num(geom.sigma_crit.value)))
st.caption(
    "Angular diameter distances: "
    + tex(r"D_\mathrm{d} = " + f"{geom.D_d.value:.0f}") + " Mpc · "
    + tex(r"D_\mathrm{s} = " + f"{geom.D_s.value:.0f}") + " Mpc · "
    + tex(r"D_\mathrm{ds} = " + f"{geom.D_ds.value:.0f}") + " Mpc · scale "
    + tex(f"{geom.kpc_per_arcsec:.2f}") + " kpc/″"
)

if real is not None:
    phys = slacs_physics(H0, Om0, Ode0, w0)
    row = phys[phys["name"] == real["name"]].iloc[0]
    r1, r2, r3, r4 = st.columns(4)
    r1.metric(r"$\sigma_\mathrm{SDSS}$ (stars)", tex(f"{row['sigma']:.0f}" + KMS) if np.isfinite(row["sigma"]) else "–")
    r2.metric(r"$\sigma_\mathrm{SIE}$ (lensing)", tex(f"{row['sigma_SIE']:.0f}" + KMS),
              help="Velocity dispersion an isothermal lens with this θ_E would need")
    if "fDM_chab" in row and np.isfinite(row["fDM_chab"]):
        r3.metric(r"dark matter inside $R_\mathrm{E}$ (Chabrier)", tex(f"{100 * row['fDM_chab']:.0f}" + PCT))
        r4.metric("… (Salpeter IMF)", tex(f"{100 * row['fDM_salp']:.0f}" + PCT))


# ---------------------------------------------------------------- cosmology curves (cached)
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


# ---------------------------------------------------------------- plots
def draw_curves(ax, curves, **kw):
    for c in curves:
        ax.plot(c[:, 0], c[:, 1], lw=0.9, **kw)


tab_names = ["Simulator", "Convergence & magnification", "Dark energy & cosmology"]
if HAVE_SLACS:
    tab_names.append("SLACS sample")
HAVE_H0LICOW = (DATA_DIR / "h0licow" / "J1206_final.csv").exists()
tab_names.append("H₀ from time delays")
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
    a1.set_title("Source plane (unlensed) + caustics")
    a2.imshow(img, origin="lower", extent=ext, cmap="inferno")
    draw_curves(a2, crit, color="#4fc3f7", ls="--")
    if len(images):
        a2.scatter(images[:, 0], images[:, 1], s=40, facecolors="none", edgecolors="#7CFC00")
    a2.plot(0, 0, "x", color="#bbb", ms=6)
    a2.set_title("Image plane (observed) + critical curves")
    for a in (a1, a2):
        a.set_xlabel(r"$x$ [arcsec]")
        a.set_ylabel(r"$y$ [arcsec]")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.latex(r"\vec\beta = \vec\theta - \vec\alpha(\vec\theta), \qquad "
             r"\mu = \left[\det \frac{\partial\vec\beta}{\partial\vec\theta}\right]^{-1}")
    if len(images):
        st.markdown(f"**{len(images)} image(s)** of the source centre (green circles):")
        table = {
            "x [″]": np.round(images[:, 0], 4),
            "y [″]": np.round(images[:, 1], 4),
            "magnification μ": np.round(images[:, 2], 3),
            "parity": np.where(images[:, 2] > 0, "+", "− (mirrored)"),
        }
        delay_col = "delay after 1st image [days]"
        try:
            dt = time_delays(lens, geom, images, bx, by)
            table[delay_col] = np.round(dt, 2)
        except NotImplementedError:
            pass
        st.dataframe(table, hide_index=True)
        if len(images) > 1 and delay_col in table:
            st.latex(r"\Delta t_{ij} = \frac{D_{\Delta t}}{c}\,\bigl[\tau(\vec\theta_i) - \tau(\vec\theta_j)\bigr],"
                     r"\qquad \tau(\vec\theta) = \tfrac12\,|\vec\theta - \vec\beta|^2 - \psi(\vec\theta)")
            st.caption("Time-delay distance " + tex(r"D_{\Delta t} = " + f"{geom.time_delay_distance().value:.0f}")
                       + r" Mpc $\propto 1/H_0$. Measuring the delays of a variable quasar therefore gives $H_0$.")
    st.caption("When the source crosses a caustic (blue, left), the number of images changes by two; "
               "the bright arcs form on the critical curves (dashed, right).")

with tab_maps:
    kap = convergence_map(lens, fov, npix)
    mu = magnification_map(lens, fov, npix)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.3))
    im1 = a1.imshow(np.log10(np.clip(kap, 1e-3, None)), origin="lower", extent=ext, cmap="viridis")
    a1.contour(*make_grid(fov, npix), kap, levels=[1.0], colors="white", linewidths=0.8)
    a1.set_title(r"$\log_{10}\kappa$  (white: $\kappa = 1$)")
    fig.colorbar(im1, ax=a1, fraction=0.046)
    im2 = a2.imshow(np.log10(np.abs(mu)), origin="lower", extent=ext, cmap="magma", vmin=-1, vmax=2)
    draw_curves(a2, crit, color="#4fc3f7", ls="--")
    a2.set_title(r"$\log_{10}|\mu|$")
    fig.colorbar(im2, ax=a2, fraction=0.046)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.latex(r"\kappa(\vec\theta) = \frac{\Sigma(\vec\theta)}{\Sigma_\mathrm{crit}}, \qquad "
             r"\Sigma_\mathrm{crit} = \frac{c^2}{4\pi G}\,\frac{D_\mathrm{s}}{D_\mathrm{d}\,D_\mathrm{ds}}")

with tab_de:
    st.markdown(
        "Cosmology enters only through the distances. This tab shows how much the **Einstein radius** "
        r"(for fixed lens physics) or the **mass inferred from an observed $\theta_\mathrm{E}$** "
        r"shifts with $\Omega_\mathrm{m}$ and $w$."
    )
    st.latex(r"M(<\theta_\mathrm{E}) = \pi\,\Sigma_\mathrm{crit}\,(D_\mathrm{d}\,\theta_\mathrm{E})^2"
             r"= \frac{c^2}{4G}\,\frac{D_\mathrm{d}\,D_\mathrm{s}}{D_\mathrm{ds}}\,\theta_\mathrm{E}^2")
    obs_theta = theta_E if np.isfinite(theta_E) else 1.0
    ws, m_w = mass_vs_w(H0, Om0, Ode0, z_d, z_s, obs_theta)
    oms, m_om = mass_vs_om(H0, None if flat else Ode0, w0, z_d, z_s, obs_theta)
    m_ref = geom.mass_in_theta_E(obs_theta)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ws, m_w / m_ref, color="#ffb74d")
    a1.axvline(w0, color="#888", ls=":")
    a1.set_xlabel(r"$w$")
    a1.set_ylabel(r"$M(<\theta_\mathrm{E})\,/\,M_\mathrm{current}$")
    a1.set_title(r"inferred mass at fixed $\theta_\mathrm{E} = " + f"{obs_theta:.3g}" + "''$")
    a2.plot(oms, m_om / m_ref, color="#4fc3f7")
    a2.axvline(Om0, color="#888", ls=":")
    a2.set_xlabel(r"$\Omega_\mathrm{m}$")
    a2.set_title(r"… as a function of $\Omega_\mathrm{m}$")
    for a in (a1, a2):
        a.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    zs, ratio = efficiency_vs_zs(H0, Om0, Ode0, w0, z_d)
    fig, a = plt.subplots(figsize=(11, 3))
    a.plot(zs, ratio, color="#ce93d8")
    a.axvline(z_s, color="#888", ls=":")
    a.set_xlabel(r"$z_\mathrm{s}$")
    a.set_ylabel(r"$D_\mathrm{ds}\,/\,D_\mathrm{s}$")
    a.set_title(r"Lensing efficiency: $\theta_\mathrm{E} \propto D_\mathrm{ds}/D_\mathrm{s}$ (SIS)"
                r" — sources at different $z$ probe cosmology")
    a.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.metric(r"time-delay distance $D_{\Delta t}$", tex(f"{geom.time_delay_distance().value:.0f}" + r"\,\mathrm{Mpc}"),
              help="D_Δt ∝ 1/H₀ -- the basis of H₀ measurements from lensed quasars.")


if HAVE_SLACS:
    with tabs[3]:
        st.markdown(
            "**63 real lens galaxies** from the Sloan Lens ACS Survey (Bolton et al. 2008; stellar masses: "
            "Auger et al. 2009), evaluated in the cosmology chosen in the sidebar. "
            r"The observed Einstein radius gives the total mass $M_\mathrm{E}$; subtracting the stellar mass "
            "leaves the dark matter."
        )
        st.latex(r"f_\mathrm{DM}(<R_\mathrm{E}) = 1 - \frac{M_\ast(<R_\mathrm{E})}{M_\mathrm{E}}, \qquad "
                 r"f_\mathrm{SIE} = \frac{\sigma_\mathrm{SDSS}}{\sigma_\mathrm{SIE}}")
        phys = slacs_physics(H0, Om0, Ode0, w0)
        good = phys["good_sigma"].to_numpy(bool)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(r"median $M_\mathrm{E}$ [$M_\odot$]", tex(num(float(np.median(phys["M_E"])))))
        k2.metric(r"median $R_\mathrm{E}$", tex(f"{np.median(phys['R_E_kpc']):.2f}" + r"\,\mathrm{kpc}"))
        k3.metric(r"$\langle \sigma_\mathrm{SDSS} / \sigma_\mathrm{SIE} \rangle$", tex(f"{phys['f_SIE'][good].mean():.3f}"),
                  help="≈ 1 means: stars and lensing see the same (isothermal) total mass")
        k4.metric(r"median $f_\mathrm{DM}$ (Chabrier)", tex(f"{100 * np.nanmedian(phys['fDM_chab']):.0f}" + PCT))

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        a1.scatter(phys["sigma"][good], phys["M_E"][good], s=14, color="#4fc3f7")
        a1.set_xscale("log")
        a1.set_yscale("log")
        a1.set_xlabel(r"$\sigma_\mathrm{SDSS}$ [km/s]")
        a1.set_ylabel(r"$M_\mathrm{E}\ [M_\odot]$")
        a1.set_title("Mass inside the Einstein radius vs. stellar velocity dispersion")
        for imf, col, lab in [("chab", "#4fc3f7", "Chabrier"), ("salp", "#ffb74d", "Salpeter")]:
            if f"fDM_{imf}" in phys:
                a2.scatter(phys["R_E_over_R_e"], phys[f"fDM_{imf}"], s=14, color=col, label=lab)
        a2.axhline(0, color="#888", lw=0.8)
        a2.set_xlabel(r"$R_\mathrm{E}\,/\,R_\mathrm{e}$")
        a2.set_ylabel(r"$f_\mathrm{DM}(<R_\mathrm{E})$")
        a2.set_title("Dark-matter fraction")
        a2.legend()
        for a in (a1, a2):
            a.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("A negative dark-matter fraction means the stellar mass would exceed the total mass, "
                   "so the Salpeter IMF is too heavy for these galaxies. This is how lenses constrain star formation.")

        cols = ["name", "z_d", "z_s", "theta_E", "sigma", "sigma_SIE", "R_E_kpc", "M_E", "ML_E", "fDM_chab", "fDM_salp"]
        show = phys[[c for c in cols if c in phys]].copy()
        show["M_E"] = show["M_E"] / 1e11
        st.dataframe(
            show.rename(columns={"theta_E": "θ_E [″]", "sigma": "σ_SDSS", "sigma_SIE": "σ_SIE",
                                 "R_E_kpc": "R_E [kpc]", "M_E": "M_E [10¹¹ M☉]", "ML_E": "M/L_V",
                                 "fDM_chab": "f_DM Chab", "fDM_salp": "f_DM Salp"}).round(3),
            hide_index=True, width="stretch",
        )


@st.cache_resource(show_spinner="Loading H0LiCOW posteriors …")
def h0licow_lenses():
    from lenscape.h0licow import load_h0licow

    return load_h0licow()


@st.cache_data(show_spinner=False)
def h0_marginals(names: tuple[str, ...], om_free: bool, om_fixed: float):
    from lenscape.h0_inference import infer_h0

    lenses = [l for l in h0licow_lenses() if l.name in names]
    H0 = np.arange(50.0, 110.0001, 0.1)
    Om = np.linspace(0.05, 0.5, 46) if om_free else np.array([om_fixed])
    single = {l.name: infer_h0([l], H0=H0, Om=Om) for l in lenses}
    joint = infer_h0(lenses, H0=H0, Om=Om)
    return (H0, {k: (v.marginal_H0(), v.summary()) for k, v in single.items()},
            (joint.marginal_H0(), joint.summary()))


def pm(m: float, lo: float, hi: float) -> str:
    """Asymmetric error in LaTeX: 73.3^{+1.7}_{-1.8}."""
    return f"{m:.1f}^{{+{hi:.1f}}}_{{-{lo:.1f}}}"


with tabs[-1]:
    st.markdown(
        "Light from the different images of a lensed quasar arrives at different times. The measured delay "
        r"and a lens model give the **time-delay distance** $D_{\Delta t} \propto 1/H_0$, independent of the "
        "distance ladder and of the early universe. Shown here are the six lenses of the **H0LiCOW** "
        "collaboration (Wong et al. 2020) with their public distance posteriors."
    )
    st.latex(r"D_{\Delta t} = (1 + z_\mathrm{d})\,\frac{D_\mathrm{d}\,D_\mathrm{s}}{D_\mathrm{ds}} \propto \frac{1}{H_0}")
    if not HAVE_H0LICOW:
        st.info("The H0LiCOW data are not there yet. Run once in the repository folder: "
                "`python scripts/fetch_h0licow.py` (~120 MB).")
    else:
        all_lenses = h0licow_lenses()
        names = [l.name for l in all_lenses]
        c1, c2 = st.columns([3, 1])
        chosen = c1.multiselect("Lenses in the combination", names, default=names)
        om_free = c2.toggle(r"$\Omega_\mathrm{m}$ free (0.05–0.5)", value=True,
                            help="Off: Ωₘ is fixed to the value from the sidebar")
        if not chosen:
            st.warning("Select at least one lens.")
        else:
            H0g, single, (pj, (m, lo, hi)) = h0_marginals(tuple(chosen), om_free, float(Om0))
            d1, d2, d3 = st.columns(3)
            d1.metric(r"$H_0$ (combined) [km s$^{-1}$ Mpc$^{-1}$]", tex(pm(m, lo, hi)))
            d2.metric(r"tension with Planck ($67.4 \pm 0.5$)",
                      tex(f"{abs(m - 67.4) / np.hypot((lo + hi) / 2, 0.5):.1f}" + r"\,\sigma"))
            d3.metric(r"tension with SH0ES ($73.04 \pm 1.04$)",
                      tex(f"{abs(m - 73.04) / np.hypot((lo + hi) / 2, 1.04):.1f}" + r"\,\sigma"))

            fig, ax = plt.subplots(figsize=(11, 4.2))
            ax.axvspan(66.9, 67.9, color="#e57373", alpha=0.25, lw=0)
            ax.axvspan(72.0, 74.08, color="#81c784", alpha=0.25, lw=0)
            ax.text(67.4, 1.02, "Planck", color="#e57373", ha="center", transform=ax.get_xaxis_transform())
            ax.text(73.04, 1.02, "SH0ES", color="#81c784", ha="center", transform=ax.get_xaxis_transform())
            for name, (p, (mm, ll, hh)) in single.items():
                ax.plot(H0g, p, lw=1, label=f"{name}: ${pm(mm, ll, hh)}$")
            ax.plot(H0g, pj, color="white", lw=2.4, label=f"combined: ${pm(m, lo, hi)}$")
            ax.set_xlim(55, 100)
            ax.set_ylim(0, None)
            ax.set_yticks([])
            ax.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
            ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#333", loc="upper right")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                r"With all six lenses and free $\Omega_\mathrm{m}$, lenscape reproduces the published result "
                r"$H_0 = 73.3^{+1.7}_{-1.8}$ km/s/Mpc. Leave out individual lenses to see how robust the tension "
                "is, e.g. without RXJ1131 or PG1115. Caveat: the result depends on assumptions about the radial "
                r"mass profile (mass-sheet degeneracy); TDCOSMO 2025 relaxes them and finds $71.6^{+3.9}_{-3.3}$."
            )
