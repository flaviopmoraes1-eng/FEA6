"""Vaso de pressão vertical: costado cilíndrico + tampo semi-elíptico 2:1 no topo, base solta,
bocal no topo (eixo Z) e bocal no costado (eixo +Y). Casca na superfície média, eixo do vaso em Z."""
import os
import tempfile
from dataclasses import dataclass

import gmsh
import numpy as np

from mesher import _LOCK

CAP_T = 50.0  # espessura das tampas planas dos bocais (mm)


@dataclass(frozen=True)
class VesselParams:
    shell_od: float
    shell_t: float
    shell_len: float  # comprimento reto do costado (base até a linha de tangência)
    head_t: float
    top_od: float
    top_t: float
    top_len: float  # projeção do bocal acima do ápice do tampo
    side_od: float
    side_t: float
    side_len: float  # projeção do bocal a partir da superfície média do costado
    top_pad_t: float = 0.0  # espessura total do reforço do bocal do topo (0 = sem reforço)
    side_pad_t: float = 0.0  # idem, bocal do costado

    @property
    def top_pad_r(self) -> float:
        return 2 * self.r_top  # raio do reforço = 2 × raio (médio) do bocal

    @property
    def side_pad_r(self) -> float:
        return 2 * self.r_side

    @property
    def side_z(self) -> float:
        return self.shell_len / 2  # bocal do costado no meio da altura do costado

    @property
    def a(self) -> float:
        return (self.shell_od - self.shell_t) / 2

    @property
    def b(self) -> float:
        return self.a / 2

    @property
    def r_top(self) -> float:
        return (self.top_od - self.top_t) / 2

    @property
    def r_side(self) -> float:
        return (self.side_od - self.side_t) / 2


def validate(p: VesselParams) -> None:
    for name in ("shell_od", "shell_t", "shell_len", "head_t", "top_od", "top_t", "top_len",
                 "side_od", "side_t", "side_len"):
        if not getattr(p, name) > 0:
            raise ValueError("Todos os diâmetros, espessuras e comprimentos devem ser maiores que 0.")
    for od, t, nome in ((p.shell_od, p.shell_t, "Costado"), (p.shell_od, p.head_t, "Tampo"),
                        (p.top_od, p.top_t, "Bocal do topo"), (p.side_od, p.side_t, "Bocal do costado")):
        if t >= od / 2:
            raise ValueError(f"{nome}: a espessura deve ser menor que metade do diâmetro externo.")
    if p.r_top >= 0.8 * p.a or p.r_side >= 0.8 * p.a:
        raise ValueError("Os bocais devem ser bem menores que o costado (raio médio < 80% do raio do vaso).")
    if p.side_z - p.r_side <= 0 or p.side_z + p.r_side >= p.shell_len:
        raise ValueError("O bocal do costado deve ficar inteiro dentro do trecho reto do costado "
                         f"(elevação entre {p.r_side:.0f} e {p.shell_len - p.r_side:.0f} mm).")
    if p.top_pad_t < 0 or p.side_pad_t < 0:
        raise ValueError("A espessura do reforço não pode ser negativa.")
    if p.top_pad_t > 0 and p.top_pad_r >= 0.9 * p.a:
        raise ValueError("Reforço do bocal do topo grande demais para o tampo (raio do reforço "
                         f"{p.top_pad_r:.0f} mm; máximo {0.9 * p.a:.0f} mm).")
    if p.side_pad_t > 0 and (p.side_pad_r >= 0.9 * p.a or p.side_z - p.side_pad_r <= 0
                             or p.side_z + p.side_pad_r >= p.shell_len):
        raise ValueError("Reforço do bocal do costado não cabe no trecho reto do costado "
                         f"(raio do reforço {p.side_pad_r:.0f} mm).")


def build_vessel_step(p: VesselParams) -> bytes:
    validate(p)
    a, b, L = p.a, p.b, p.shell_len
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
        path = f.name
    with _LOCK:
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("vessel")
            occ = gmsh.model.occ
            circle = occ.addCircle(0, 0, 0, a)
            cyl = occ.extrude([(1, circle)], 0, 0, L)[1][1]
            arc = occ.addEllipse(0, 0, L, a, b, angle1=0, angle2=np.pi / 2,
                                 zAxis=[0, -1, 0], xAxis=[1, 0, 0])
            head = occ.revolve([(1, arc)], 0, 0, 0, 0, 0, 1, 2 * np.pi)
            head = [t for d, t in head if d == 2][0]
            apex = L + b
            c_top = occ.addCircle(0, 0, L, p.r_top)
            top_out = occ.extrude([(1, c_top)], 0, 0, apex + p.top_len - L)
            top = top_out[1][1]
            top_cap = occ.addPlaneSurface([occ.addCurveLoop([top_out[0][1]])])
            # Bocal do costado ao longo de +Y (a emenda do cilindro fica em +X)
            c_side = occ.addCircle(0, 0, p.side_z, p.r_side, zAxis=[0, 1, 0], xAxis=[1, 0, 0])
            side_out = occ.extrude([(1, c_side)], 0, a + p.side_len, 0)
            side = side_out[1][1]
            cap = occ.addPlaneSurface([occ.addCurveLoop([side_out[0][1]])])
            tools = [(2, top), (2, side), (2, cap), (2, top_cap)]
            # Cilindros auxiliares que recortam a região do reforço nas paredes (removidos depois)
            if p.top_pad_t > 0:
                c = occ.addCircle(0, 0, L, p.top_pad_r)
                tools.append((2, occ.extrude([(1, c)], 0, 0, b + 1)[1][1]))
            if p.side_pad_t > 0:
                c = occ.addCircle(0, 0, p.side_z, p.side_pad_r, zAxis=[0, 1, 0], xAxis=[1, 0, 0])
                tools.append((2, occ.extrude([(1, c)], 0, a + 1, 0)[1][1]))
            _, outmap = occ.fragment([(2, cyl), (2, head)], tools)
            occ.synchronize()

            def com(tag):
                return np.array(occ.getCenterOfMass(2, tag))

            def extent(tag):
                x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, tag)
                return np.array([x1 - x0, y1 - y0, z1 - z0])

            kill = [t for m in outmap[6:] for _, t in m]
            # Trecho interno de cada bocal: a peça cujo centro fica mais perto do interior do vaso
            kill.append(min((t for _, t in outmap[2]), key=lambda t: com(t)[2]))
            kill.append(min((t for _, t in outmap[3]), key=lambda t: com(t)[1]))
            # Aberturas nas paredes do vaso, sob cada bocal
            for _, t in outmap[1]:
                e = extent(t)
                if e[0] <= 2.02 * p.r_top and e[1] <= 2.02 * p.r_top:
                    kill.append(t)
            for _, t in outmap[0]:
                e = extent(t)
                if e[0] <= 2.02 * p.r_side and e[2] <= 2.02 * p.r_side:
                    kill.append(t)
            occ.remove([(2, t) for t in set(kill)])
            occ.synchronize()
            gmsh.write(path)
        finally:
            gmsh.finalize()
    try:
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)


def element_thickness(nodes: np.ndarray, tri: np.ndarray, p: VesselParams) -> np.ndarray:
    """Espessura por triângulo: bocais (todos os vértices na superfície do bocal), tampo (z > L)
    e costado."""
    L = p.shell_len
    pts = nodes[tri]  # (M, 3, 3)
    t = np.where(pts[:, :, 2].mean(axis=1) > L, p.head_t, p.shell_t)
    rho = np.hypot(pts[:, :, 0], pts[:, :, 1])
    on_top = (np.abs(rho - p.r_top) < 2e-3 * p.r_top + 1e-6).all(axis=1) & (pts[:, :, 2] > L).all(axis=1)
    d_side = np.hypot(pts[:, :, 0], pts[:, :, 2] - p.side_z)
    on_side = (np.abs(d_side - p.r_side) < 2e-3 * p.r_side + 1e-6).all(axis=1) & (pts[:, :, 1] > 0).all(axis=1)
    t = np.where(on_top, p.top_t, t)
    t = np.where(on_side, p.side_t, t)
    if p.top_pad_t > 0:
        c = pts.mean(axis=1)
        in_pad = (np.hypot(c[:, 0], c[:, 1]) < p.top_pad_r * 1.001) & (c[:, 2] > L) & ~on_top
        t = np.where(in_pad, p.top_pad_t, t)
    if p.side_pad_t > 0:
        c = pts.mean(axis=1)
        in_pad = (np.hypot(c[:, 0], c[:, 2] - p.side_z) < p.side_pad_r * 1.001) & (c[:, 1] > 0) & ~on_side
        t = np.where(in_pad, p.side_pad_t, t)
    z_cap = L + p.b + p.top_len
    y_cap = p.a + p.side_len
    is_cap = (np.abs(pts[:, :, 2] - z_cap) < 1e-6 * z_cap + 1e-6).all(axis=1) | \
        (np.abs(pts[:, :, 1] - y_cap) < 1e-6 * y_cap + 1e-6).all(axis=1)
    return np.where(is_cap, CAP_T, t)
