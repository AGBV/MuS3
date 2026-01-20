import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from skimage import color, data, img_as_float

# --- Konfiguration der Seite ---
st.set_page_config(page_title="MuS³ - Bildrestaurierung", layout="wide")

st.title("MuS³ Übung 4: Inverse Filterung & Wiener Filter")
st.markdown("""
Diese interaktive App demonstriert die Auswirkung der **Point Spread Function (PSF)**,
des **Rauschens** und verschiedener **Restaurationstechniken** (Inverse, Pseudo-Inverse, Wiener, Unscharfmaskierung).
""")

# --- Sidebar: Parameter ---
st.sidebar.header("1. Bildquelle")

# --- Vereinfachte Upload Logik ---
# Der Uploader ist immer da.
uploaded_file = st.sidebar.file_uploader(
    "Bild hochladen (optional, sonst Cameraman)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

raw_image = None

if uploaded_file is not None:
    try:
        # Bild mit PIL laden
        pil_image = Image.open(uploaded_file)
        raw_image = np.array(pil_image)

        # Falls Farbbild (3 Kanäle RGB), in Graustufen wandeln
        if raw_image.ndim == 3 and raw_image.shape[2] in [3, 4]:
            # Slice [: , :, :3] ignoriert einen eventuellen Alpha-Kanal
            raw_image = color.rgb2gray(raw_image[:, :, :3])

        st.sidebar.success("Eigenes Bild geladen!")

    except Exception as e:
        st.sidebar.error(f"Fehler beim Laden: {e}")
        raw_image = None  # Fallback auslösen bei Fehler

# Fallback: Wenn kein Upload oder Fehler beim Upload -> Cameraman
if raw_image is None:
    st.sidebar.info("Verwende Standardbild (Cameraman).")
    raw_image = data.camera()

# Sicherstellen, dass Bild als Float [0, 1] vorliegt
image = img_as_float(raw_image)


# --- Sidebar: System Parameter ---
st.sidebar.markdown("---")
st.sidebar.header("2. Systemparameter")

# PSF Parameter
sigma_psf = st.sidebar.slider("PSF Sigma (Unschärfe)", 0.5, 5.0, 1.0, 0.1)
kernel_size = st.sidebar.slider("PSF Kernel Größe", 3, 15, 5, 2)

# Rausch Parameter
noise_var = st.sidebar.slider(
    "Rausch-Varianz", 0.0, 0.01, 0.0001, 0.0001, format="%.4f"
)


# --- Hilfsfunktionen ---


def gaussian_kernel(size, sigma):
    """Erzeugt einen 2D Gauss-Kernel."""
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def get_mtf(psf):
    """Berechnet die MTF für Visualisierung (feste Größe)."""
    mtf = np.abs(np.fft.fftshift(np.fft.fft2(psf, s=(64, 64))))
    return mtf


# --- 1. PSF und MTF Visualisierung ---
st.header("1. Analyse der Point Spread Function (PSF)")

psf = gaussian_kernel(kernel_size, sigma_psf)
mtf_vis = get_mtf(psf)

col1, col2 = st.columns(2)

with col1:
    st.subheader("PSF (Ortsbereich)")
    x = np.arange(psf.shape[0])
    y = np.arange(psf.shape[1])
    fig_psf = go.Figure(data=[go.Surface(z=psf, x=x, y=y, colorscale="Viridis")])
    fig_psf.update_layout(
        title="PSF 3D",
        autosize=False,
        width=400,
        height=400,
        margin=dict(l=0, r=0, b=0, t=30),
    )
    st.plotly_chart(fig_psf)

with col2:
    st.subheader("MTF (Frequenzbereich)")
    fig_mtf = go.Figure(data=[go.Surface(z=mtf_vis, colorscale="Jet")])
    fig_mtf.update_layout(
        title="MTF 3D",
        autosize=False,
        width=400,
        height=400,
        margin=dict(l=0, r=0, b=0, t=30),
    )
    st.plotly_chart(fig_mtf)


# --- 2. Bildverarbeitung (Faltung & Rauschen) ---
st.header("2. Bilddegradation")

# Padding Größen berechnen
img_h, img_w = image.shape
psf_h, psf_w = psf.shape
fft_shape = (img_h + psf_h - 1, img_w + psf_w - 1)

# Frequenzbereich Transformationen
fft_I = np.fft.fft2(image, s=fft_shape)
fft_PSF = np.fft.fft2(psf, s=fft_shape)

# Faltung
fft_I_blur = fft_I * fft_PSF
I_blur = np.abs(np.fft.ifft2(fft_I_blur))

# Rauschen hinzufügen
np.random.seed(42)
noise = np.random.normal(0, np.sqrt(noise_var), I_blur.shape)
I_blur_noise = I_blur + noise
I_blur_noise = np.clip(I_blur_noise, 0, 1)

# FFT des verrauschten Bildes neu berechnen für Filterung
fft_I_blur_noise = np.fft.fft2(I_blur_noise, s=fft_shape)
fft_PSF_abs = np.abs(fft_PSF)

# Anzeige Original vs Degradiert
col_orig, col_blur, col_noise = st.columns(3)
with col_orig:
    st.image(image, caption="Original", width="stretch", clamp=True)
with col_blur:
    st.image(
        I_blur[:img_h, :img_w],
        caption="Nur Unschärfe (Faltung)",
        width="stretch",
        clamp=True,
    )
with col_noise:
    st.image(
        I_blur_noise[:img_h, :img_w],
        caption=f"Unscharf + Rauschen (Var: {noise_var})",
        width="stretch",
        clamp=True,
    )


# --- 3. Bildrestaurierung ---
st.header("3. Restaurierungsmethoden")
st.markdown("Wählen Sie Parameter für die verschiedenen Filtertypen.")

tabs = st.tabs(
    ["Inverser Filter", "Pseudo-Invers", "Wiener Filter", "Unscharfmaskierung"]
)

# --- TAB 1: Inverser Filter ---
with tabs[0]:
    st.markdown(
        "Der **inverse Filter** dividiert einfach durch die Übertragungsfunktion."
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        fft_inv = 1.0 / fft_PSF_abs
        fft_restored_inv = fft_I_blur_noise * fft_inv

    I_restored_inv = np.abs(np.fft.ifft2(fft_restored_inv))

    st.image(
        I_restored_inv[:img_h, :img_w],
        caption="Ergebnis Inverser Filter",
        width="stretch",
        clamp=True,
    )

# --- TAB 2: Pseudo-Inverser Filter ---
with tabs[1]:
    st.markdown(
        "Der **Pseudo-Inverse Filter** begrenzt die Verstärkung bei kleinen Werten von H."
    )

    threshold = st.slider(
        "Schwellwert (Threshold)", 1.0, 100.0, 40.0, 1.0, key="pseudo_thr"
    )
    method = st.radio(
        "Methode bei Überschreitung",
        ["Zu Null setzen", "Konstant halten (Clipping)"],
        index=0,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        fft_inv_pseudo = 1.0 / fft_PSF_abs

    if method == "Zu Null setzen":
        fft_inv_pseudo[fft_inv_pseudo > threshold] = 0
    else:
        fft_inv_pseudo[fft_inv_pseudo > threshold] = threshold

    fft_restored_pseudo = fft_I_blur_noise * fft_inv_pseudo
    I_restored_pseudo = np.abs(np.fft.ifft2(fft_restored_pseudo))

    st.image(
        I_restored_pseudo[:img_h, :img_w],
        caption=f"Pseudo-Invers (Thr={threshold})",
        width="stretch",
        clamp=True,
    )

# --- TAB 3: Wiener Filter ---
with tabs[2]:
    st.markdown(
        r"Der **Wiener Filter** berücksichtigt das Verhältnis von Rausch- zu Signalleistung."
    )

    k_wiener = st.slider("Faktor K (Sn/Si)", 0.0, 0.2, 0.01, 0.001, format="%.4f")

    # Wiener Formel
    fft_wiener_filter = np.conj(fft_PSF) / ((np.abs(fft_PSF) ** 2) + k_wiener)

    fft_restored_wiener = fft_I_blur_noise * fft_wiener_filter
    I_restored_wiener = np.abs(np.fft.ifft2(fft_restored_wiener))

    st.image(
        I_restored_wiener[:img_h, :img_w],
        caption=f"Wiener Filter (K={k_wiener})",
        width="stretch",
        clamp=True,
    )

# --- TAB 4: Unscharfmaskierung ---
with tabs[3]:
    st.markdown("Die **Unscharfmaskierung** verstärkt hohe Frequenzen.")

    k_unsharp = st.slider("Verstärkungsfaktor k", 0.0, 2.0, 0.7, 0.1)

    mag_I_blur_noise = np.abs(fft_I_blur_noise)
    phase_I_blur_noise = np.angle(fft_I_blur_noise)

    high_freq_part = mag_I_blur_noise - np.abs(fft_I_blur_noise * np.abs(fft_PSF))
    new_magnitude = (k_unsharp * high_freq_part) + mag_I_blur_noise

    fft_unsharp = new_magnitude * np.exp(1j * phase_I_blur_noise)
    I_unsharp = np.abs(np.fft.ifft2(fft_unsharp))

    st.image(
        I_unsharp[:img_h, :img_w],
        caption=f"Unscharfmaskierung (k={k_unsharp})",
        width="stretch",
        clamp=True,
    )
