import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d



def gauss(imagen: np.ndarray, desv_est):

    medio_largo = np.ceil(2 * desv_est)

    y, x = np.ogrid[-medio_largo:medio_largo + 1 , -medio_largo:medio_largo + 1]

    kernel = np.exp(-(x**2 + y**2)/(2 * desv_est **2))

    kernel_norm =  kernel/np.sum(kernel)

    return convolve2d(imagen, kernel_norm, mode="same", boundary="symm")


def a_pantalla(imagen, gamma=3.2):

    positivo = np.maximum(imagen, 0.0)

    imagen_norm = positivo / (np.max(positivo) + 1e-12)

    return np.clip(imagen_norm, 0.0, 1.0) ** (1.0 / gamma)

def piramide_laplaciana(imagen_original, desv_est, niveles=5, ):
    piramide_laplaciana = []
    imagen_nivel_act = imagen_original

    for _ in range(niveles):
        imagen_suave = gauss(imagen_nivel_act, desv_est)
        imagen_suave_mitad = imagen_suave[::2, ::2]


        alto, ancho = imagen_nivel_act.shape
        imagen_suave_reexp = cv2.resize(imagen_suave_mitad, (ancho, alto), interpolation=cv2.INTER_LINEAR)

        imagen_laplaciana = imagen_nivel_act - imagen_suave_reexp

        piramide_laplaciana.append(imagen_laplaciana)

        imagen_nivel_act = imagen_suave_mitad

    return piramide_laplaciana, imagen_nivel_act


def starlet(imagen_original, niveles=5):
    capas = []
    c = imagen_original.astype(np.float64)
    B3 = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0

    for j in range(niveles):

        h = np.zeros(4 * 2**j + 1)
        h[::2**j] = B3      

        c_sig = convolve1d(convolve1d(c, h, axis=0, mode="mirror"), h, axis=1, mode="mirror")

        capas.append(c - c_sig)
        c = c_sig

    return capas, c



def reconstruccion(piramide, residuo, modo: str):

    if modo == "Laplaciana":
        imagen_rec = residuo.copy()

        for capa_detalle in piramide[::-1]:
            alto, ancho = capa_detalle.shape

            expandida = cv2.resize(imagen_rec, (ancho, alto), interpolation=cv2.INTER_LINEAR)
            imagen_rec = capa_detalle + expandida

        return imagen_rec
    
    if modo == "Starlet":
        imagen_rec = residuo.copy()

        for capa_detalle in piramide[::-1]:
            imagen_rec += capa_detalle

        return imagen_rec




def rmse(img_original: np.ndarray, img_reconstruida: np.ndarray) -> float:
    return float(np.sqrt(np.mean((img_original - img_reconstruida) ** 2)))


def mostrar_piramide(piramide: list, residuo: np.ndarray, modo: str, desv_est: float, vlim=None):

    capas_detalle = [capa for capa in piramide]
    n_capas = len(piramide)

    if vlim is None:
        v_global = float(max(np.percentile(np.abs(capa), 99.5) for capa in piramide))
        vlims = [v_global] * n_capas
        texto_escala = f"Escala global: [±{v_global:.3f}]"
    else:        
        vlims = [float(v) for v in vlim]
        texto_escala = "Escala compartida por capas"

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for i, imagen_nivel in enumerate(capas_detalle):
        lim = vlims[i]
        im = axes[i].imshow(imagen_nivel, cmap="gray", vmin=-lim, vmax=lim)
        axes[i].set_title(f"Detalle d{i+1} | {imagen_nivel.shape[0]}×{imagen_nivel.shape[1]}")
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)


    im_res = axes[5].imshow(residuo, cmap="gray", vmin=0.0, vmax=1.0)
    axes[5].set_title(f"Residual c{len(capas_detalle)} | {residuo.shape[0]}×{residuo.shape[1]}")
    axes[5].axis("off")
    fig.colorbar(im_res, ax=axes[5], fraction=0.046, pad=0.04)


    if modo == "Laplaciana":

        fig.suptitle(
            f"Pirámide Laplaciana (σ={desv_est}, interpolación bilineal) — {texto_escala}",
            fontsize=13
        )

    else:

        fig.suptitle(
            f"Transformada Starlet (B3-spline h, borde espejo) — {texto_escala}",
            fontsize=13
        )

    plt.tight_layout()
    
def comparar_reconstrucciones(imagenes: list, titulos: list, titulo_general: str, filas: int = 1, cols: int = 3):
    fig, axes = plt.subplots(filas, cols, figsize=(5 * cols, 4.5 * filas))
    axes = np.atleast_1d(axes).ravel()

    for ax, img, titulo in zip(axes, imagenes, titulos):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(titulo)
        ax.axis("off")

    fig.suptitle(titulo_general, fontsize=13)
    plt.tight_layout()

def umbral_duro(piramide: list, alpha: float):

    capas_umb = []
    porcentajes = []

    for capa in piramide:
        T_j = alpha * np.max(np.abs(capa))
        mascara = np.abs(capa) >= T_j

        capas_umb.append(capa * mascara)

        porcentajes.append(100 * np.count_nonzero(mascara) / mascara.size)

    return capas_umb, porcentajes


def aplicar_ganancias(piramide: list, factores: list) :

    return [factor * capa for factor, capa in zip(factores, piramide)]



desv_est = 2.0
alpha = 0.5
factores_finos = [2.0, 2.0, 1.5, 0.8, 0.5]
factores_gruesos = [0.7, 0.8, 1.2, 1.7, 2.3]


#==========================================================Parte 0 (implementacion necesaria)================================================================================================

imagen = cv2.imread("archivos_t4/belgium.hdr", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
imagen_l = 0.177 * imagen_rgb[..., 0] + 0.813 * imagen_rgb[..., 1] + 0.011 * imagen_rgb[..., 2]
imagen_adaptada_32 = a_pantalla(imagen_l, gamma=3.2)

piramide_lap, residuo_l = piramide_laplaciana(imagen_adaptada_32, desv_est)
piramide_star, residuo_s = starlet(imagen_adaptada_32 )

imagen_reconstruida_l = reconstruccion(piramide_lap, residuo_l, "Laplaciana")
imagen_reconstruida_s = reconstruccion(piramide_star, residuo_s, 'Starlet')

vlims_por_capa = [
    max(float(np.percentile(np.abs(capa_l), 99.5)),
        float(np.percentile(np.abs(capa_s), 99.5)),)
    for capa_l, capa_s in zip(piramide_lap, piramide_star)]


#==========================================================Parte 1================================================================================================

mostrar_piramide(piramide_lap, residuo_l,  'Laplaciana', desv_est, vlims_por_capa)
mostrar_piramide(piramide_star, residuo_s, 'Starlet', desv_est, vlims_por_capa)

print(rmse(imagen_adaptada_32, imagen_reconstruida_s))
print(rmse(imagen_adaptada_32, imagen_reconstruida_l))

comparar_reconstrucciones(
     [imagen_adaptada_32, imagen_reconstruida_l, imagen_reconstruida_s],
     ["Original (γ=3.2)", "Reconstruida Laplaciana", "Reconstruida Starlet"],
     "Verificación de reconstrucción sin modificar coeficientes",
     filas=1, cols=3
 )
#==========================================================Parte 2================================================================================================

piramide_lap_umb, porcentaje_coef_l = umbral_duro(piramide_lap, alpha)
piramide_star_umb, porcentaje_coef_s = umbral_duro(piramide_star, alpha)

mostrar_piramide(piramide_lap_umb, residuo_l,  'Laplaciana', desv_est, vlims_por_capa)
mostrar_piramide(piramide_star_umb, residuo_s, 'Starlet', desv_est, vlims_por_capa)

imagen_reconstruida_l_umb = reconstruccion(piramide_lap_umb, residuo_l, "Laplaciana")
imagen_reconstruida_s_umb = reconstruccion(piramide_star_umb, residuo_s, 'Starlet')

print(f"α={alpha} | % conservado Lap: {np.round(porcentaje_coef_l, 2)} | Starlet: {np.round(porcentaje_coef_s, 2)}")
print(f"      RMSE vs original: Lap {rmse(imagen_adaptada_32, imagen_reconstruida_l_umb):.4f} | Starlet {rmse(imagen_adaptada_32, imagen_reconstruida_s_umb):.4f}")

comparar_reconstrucciones(
     [imagen_adaptada_32, imagen_reconstruida_l_umb, imagen_reconstruida_s_umb],
     ["Original", f"Laplaciana (α={alpha})", f"Starlet (α={alpha})"],
     f"Reconstrucción tras umbral duro (α={alpha})",
     filas=1, cols=3
 )

#==========================================================Parte 3================================================================================================

piramide_fina_l = aplicar_ganancias(piramide_lap, factores_finos)
piramide_fina_s = aplicar_ganancias(piramide_star, factores_finos)

piramide_gruesa_l = aplicar_ganancias(piramide_lap, factores_gruesos)
piramide_gruesa_s = aplicar_ganancias(piramide_star, factores_gruesos)

mostrar_piramide(piramide_fina_l, residuo_l,  'Laplaciana', desv_est, vlims_por_capa)
mostrar_piramide(piramide_fina_s, residuo_s, 'Starlet', desv_est, vlims_por_capa)
mostrar_piramide(piramide_gruesa_l, residuo_l,  'Laplaciana', desv_est, vlims_por_capa)
mostrar_piramide(piramide_gruesa_s, residuo_s, 'Starlet', desv_est, vlims_por_capa)

reconstruida_l_fina = reconstruccion(piramide_fina_l, residuo_l, "Laplaciana")
reconstruida_s_fina = reconstruccion(piramide_fina_s, residuo_s, 'Starlet')
reconstruida_l_gruesa = reconstruccion(piramide_gruesa_l, residuo_l, "Laplaciana")
reconstruida_s_gruesa = reconstruccion(piramide_gruesa_s, residuo_s, 'Starlet')

print(f"      RMSE vs original (fina): Lap {rmse(imagen_adaptada_32, reconstruida_l_fina):.4f} | Starlet {rmse(imagen_adaptada_32, reconstruida_s_fina):.4f}")
print(f"      RMSE vs original (gruesa): Lap {rmse(imagen_adaptada_32, reconstruida_l_gruesa):.4f} | Starlet {rmse(imagen_adaptada_32, reconstruida_s_gruesa ):.4f}")

comparar_reconstrucciones(
     [reconstruida_l_fina, reconstruida_s_fina, reconstruida_l_gruesa, reconstruida_s_gruesa],
     [
         f"Laplaciana — Finas {factores_finos}",
         f"Starlet — Finas {factores_finos}",
         f"Laplaciana — Gruesas {factores_gruesos}",
         f"Starlet — Gruesas {factores_gruesos}",
     ],
     "Modificación por ganancias (escalas finas vs. gruesas)",
     filas=2, cols=2
)

plt.show()