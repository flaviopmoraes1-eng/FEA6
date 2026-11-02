import hashlib
import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

import fea
import isometric
import report
import vessel
from mesher import free_curves, mesh_step_shell
from viewer import mesh_viewer

DOFS = ["UX", "UY", "UZ", "RX", "RY", "RZ"]
DOF_LABELS = {"UX": "UX (translação X)", "UY": "UY (translação Y)", "UZ": "UZ (translação Z)",
              "RX": "RX (rotação X)", "RY": "RY (rotação Y)", "RZ": "RZ (rotação Z)"}

st.set_page_config(page_title="Análise de defeitos API-579 com FEA", layout="wide")
st.title("ANÁLISE DE DEFEITOS TIPO TRINCA E PERDA DE ESPESSURA PELA API-579 COM FEA")
st.caption("Geometria (STEP/STL) → malha de casca → condições de contorno e cargas")

with st.sidebar:
    uploaded = st.file_uploader("Geometria (STEP, STL ou isométrico PDF)",
                                type=["step", "stp", "stl", "pdf"])
    is_pdf = uploaded is not None and uploaded.name.lower().endswith(".pdf")
    shell_default = is_pdf or (uploaded is None and st.session_state.get("manual") is not None)
    divisions = st.slider("Densidade da malha - STEP (divisões da diagonal)", 5, 100,
                          60 if shell_default else 20)
    refine = st.checkbox("Refinar malha")
    refinements = st.slider("Níveis de refinamento", 1, 3, 1) if refine else 0
    show_mesh = st.checkbox("Mostrar malha (arestas dos elementos)", value=False)
    show_flags = st.checkbox("Mostrar bandeirinhas no 3D", value=True)
    len_scale = st.number_input(
        "Fator de escala dos comprimentos", value=1.0, min_value=1e-9, format="%.6g",
        help="Multiplica as medidas numéricas do modelo (comprimentos, análise); a forma "
             "da geometria não muda. Ex.: 1000 se o arquivo está em m e você quer mm.")


@st.cache_data(show_spinner="Gerando malha...")
def build(data: bytes, suffix: str, divisions: int, refinements: int):
    return mesh_step_shell(data, divisions=divisions, refinements=refinements,
                           suffix=suffix)


def is_anchor(dofs: dict) -> bool:
    return all(dofs.get(d) == 0 for d in DOFS)


def edge_form(selected: list[int], bc: dict, lengths: dict):
    lens = [lengths.get(t, 0.0) for t in selected]
    st.markdown(f"**Arestas selecionadas:** {', '.join(map(str, selected))}")
    if len(selected) == 1:
        st.markdown(f"**Comprimento:** {lens[0]:.2f} mm")
    else:
        st.markdown("**Comprimentos:** " + "; ".join(
            f"#{t}: {l:.2f} mm" for t, l in zip(selected, lens)) + f" — **total {sum(lens):.2f} mm**")
    mode = st.radio("Restrição", ["Ancoragem (6 GDL = 0)", "Personalizada"], key="edge_mode")
    dofs = {}
    if mode.startswith("Ancoragem"):
        dofs = {d: 0.0 for d in DOFS}
    else:
        st.caption("Marque os GDL restritos e informe o valor imposto (0 = fixo). "
                   "Translações na unidade de comprimento do modelo; rotações em rad.")
        for d in DOFS:
            c1, c2 = st.columns([1.2, 1])
            on = c1.checkbox(DOF_LABELS[d], key=f"dof_on_{d}")
            val = c2.number_input("valor", value=0.0, key=f"dof_val_{d}",
                                  label_visibility="collapsed", disabled=not on,
                                  format="%.6g")
            if on:
                dofs[d] = float(val)
    c1, c2 = st.columns(2)
    if c1.button("Aplicar", key="apply_edge", type="primary", disabled=not dofs):
        for tag in selected:
            bc["curves"][tag] = dict(dofs)
        st.rerun()
    if c2.button("Remover", key="clear_edge"):
        for tag in selected:
            bc["curves"].pop(tag, None)
        st.rerun()


def edge_load_form(selected: list[int], bc: dict, lengths: dict):
    st.markdown("**Força nas arestas (total)**")
    st.caption("Força total em N nas direções globais X, Y e Z, distribuída uniformemente ao longo "
               "da(s) aresta(s) selecionada(s). Com várias arestas, o total é repartido "
               "proporcionalmente ao comprimento de cada uma.")
    cur = [sum(bc["edge_forces"].get(t, [0.0] * 3)[d] for t in selected) for d in range(3)]
    sig = "_".join(map(str, selected))
    total = [st.number_input(f"F{ax} total (N)", value=float(v), format="%.6g",
                             key=f"edge_F_{ax}_{sig}_{v}")
             for ax, v in zip("XYZ", cur)]
    c1, c2 = st.columns(2)
    if c1.button("Aplicar", key="apply_edge_q", type="primary", disabled=not any(total)):
        ltot = sum(lengths.get(t, 0.0) for t in selected)
        for tag in selected:
            share = lengths.get(tag, 0.0) / ltot if ltot > 0 else 1 / len(selected)
            bc["edge_forces"][tag] = [float(c) * share for c in total]
        st.rerun()
    if c2.button("Remover", key="clear_edge_q",
                 disabled=not any(t in bc["edge_forces"] for t in selected)):
        for tag in selected:
            bc["edge_forces"].pop(tag, None)
        st.rerun()


def crack_stress(mesh, res, tag: int):
    """Tensões normais à aresta {"mid","top","bottom"} do resultado atual (None sem resultado)."""
    if res is None or not res.solved or res.local is None:
        return None
    return fea.normal_stress_to_edge(mesh.nodes, mesh.triangles, mesh.curves[tag], res.local)


def crack_form(selected: list[int], bc: dict, mesh, res):
    st.markdown("**Trinca**")
    kind = st.radio("Tipo de trinca", ["Superficial", "Embebida"], horizontal=True, key="crack_kind")
    c1, c2 = st.columns(2)
    a = c1.number_input("Altura a (mm)", min_value=0.0, value=1.0, format="%.4g", key="crack_a")
    c2l = c2.number_input("Comprimento 2c (mm)", min_value=0.0, value=6.0, format="%.4g", key="crack_2c")
    p = None
    if kind == "Embebida":
        p = st.number_input("Distância mínima até a superfície p (mm)", min_value=0.0, value=1.0,
                            format="%.4g", key="crack_p")
    b1, b2 = st.columns(2)
    if b1.button("Adicionar trinca", key="apply_crack", type="primary"):
        for tag in selected:
            nid = max([c["id"] for c in bc["cracks"]], default=0) + 1
            bc["cracks"].append({"id": nid, "edge": int(tag), "type": kind, "a": float(a),
                                 "2c": float(c2l), "p": None if p is None else float(p),
                                 "stress": crack_stress(mesh, res, tag)})
        st.rerun()
    if b2.button("Remover trinca", key="clear_crack",
                 disabled=not any(c["edge"] in selected for c in bc["cracks"])):
        bc["cracks"] = [c for c in bc["cracks"] if c["edge"] not in selected]
        st.rerun()


def crack_text(c: dict) -> str:
    s = f"{c['type']}: a = {c['a']:g} mm, 2c = {c['2c']:g} mm"
    return s + (f", p = {c['p']:g} mm" if c.get("p") is not None else "")

def surface_form(selected: list[int], bc: dict, all_surfaces: list[int]):
    st.markdown(f"**Superfícies selecionadas:** {', '.join(map(str, selected))}")
    p = st.number_input("Pressão (unidades consistentes, ex.: MPa com mm/N)",
                        value=0.0, format="%.6g", key="pressure_val",
                        help="Positiva = atua de fora para dentro (contrária à normal externa, "
                             "pressão externa). Negativa = pressão interna.")
    c1, c2 = st.columns(2)
    if c1.button("Aplicar", key="apply_surf", type="primary"):
        for tag in selected:
            bc["pressure"][tag] = float(p)
        st.session_state.recalc = bool(p) and bool(bc["curves"])
        st.rerun()
    if c2.button("Remover", key="clear_surf"):
        for tag in selected:
            bc["pressure"].pop(tag, None)
        st.rerun()
    if st.button("Aplicar em todas as áreas", key="apply_surf_all",
                 help="Aplica a pressão acima a todas as superfícies do modelo."):
        for tag in all_surfaces:
            bc["pressure"][tag] = float(p)
        st.session_state.recalc = bool(p) and bool(bc["curves"])
        st.rerun()


def corrosion_form(node, bc: dict, mesh, nominal: float):
    st.caption("Área corroída: círculo de raio R (medido sobre a superfície) centrado no nó, "
               "com espessura reduzida. Os elementos dentro da área usam a espessura corroída.")
    if node is None:
        st.info("Clique em um nó do modelo para definir o centro da corrosão.")
    else:
        x, y, z = mesh.nodes[node]
        st.markdown(f"**Nó selecionado:** {node}  \n({x:.1f}, {y:.1f}, {z:.1f}) mm")
    default_t = round(0.5 * nominal, 6)
    if st.session_state.get("corr_t", default_t) >= nominal:
        st.session_state.corr_t = default_t
    radius = st.number_input(
        "Raio da área corroída (mm)",
        min_value=1e-6, value=100.0, step=5.0, key="corr_radius")
    t_corr = st.number_input(
        f"Espessura corroída (nominal = {nominal:g})", min_value=1e-6, max_value=float(nominal) * 0.999,
        value=default_t, format="%.6g", key="corr_t")
    c1, c2 = st.columns(2)
    if c1.button("Adicionar", key="add_corr", type="primary", disabled=node is None):
        bc["corrosion"].append({"node": int(node), "radius": float(radius), "t": float(t_corr)})
        st.rerun()
    if c2.button("Remover todas", key="clear_corr", disabled=not bc["corrosion"]):
        bc["corrosion"].clear()
        st.rerun()


def force_form(node, bc: dict, mesh):
    st.caption("Força concentrada no nó, em N, nas direções globais X, Y e Z. "
               "Uma nova força no mesmo nó substitui a anterior.")
    current = next((l for l in bc["forces"] if l["node"] == node), None)
    if node is None:
        st.info("Clique em um nó do modelo para aplicar a força.")
    else:
        x, y, z = mesh.nodes[node]
        st.markdown(f"**Nó selecionado:** {node}  \n({x:.1f}, {y:.1f}, {z:.1f}) mm")
    default = current["F"] if current else [0.0, 0.0, 0.0]
    comps = [st.number_input(f"F{ax} (N)", value=float(v), format="%.6g",
                             key=f"force_{ax}_{node}_{v}")
             for ax, v in zip("XYZ", default)]
    c1, c2 = st.columns(2)
    if c1.button("Aplicar", key="apply_force", type="primary",
                 disabled=node is None or not any(comps)):
        bc["forces"] = [l for l in bc["forces"] if l["node"] != node]
        bc["forces"].append({"node": int(node), "F": [float(c) for c in comps]})
        st.rerun()
    if c2.button("Remover", key="clear_force", disabled=current is None):
        bc["forces"] = [l for l in bc["forces"] if l["node"] != node]
        st.rerun()


def summary(bc: dict, nominal: float, sig: str):
    """Editable table of assigned conditions; any applied edit triggers a new analysis."""
    nan = float("nan")
    cols = ["Pressão", "FX (N)", "FY (N)", "FZ (N)", "Raio (mm)", "Espessura corroída"] + DOFS
    rows, refs = [], []

    def add(kind, ident, **vals):
        rows.append({"Tipo": kind, "ID": ident, **{c: vals.get(c, nan) for c in cols}})
        refs.append(kind)

    for tag, dofs in sorted(bc["curves"].items()):
        add("Aresta", tag, **{d: dofs.get(d, nan) for d in DOFS})
        refs[-1] = ("curve", tag)
    for tag, q in sorted(bc["edge_forces"].items()):
        add("Força total (aresta)", tag, **dict(zip(["FX (N)", "FY (N)", "FZ (N)"], q)))
        refs[-1] = ("edge_force", tag)
    for tag, p in sorted(bc["pressure"].items()):
        add("Superfície", tag, **{"Pressão": p})
        refs[-1] = ("pressure", tag)
    for i, l in enumerate(bc["forces"]):
        add("Força (nó)", l["node"], **dict(zip(["FX (N)", "FY (N)", "FZ (N)"], l["F"])))
        refs[-1] = ("force", i)
    for i, d in enumerate(bc["corrosion"]):
        add(f"LTA {i + 1}", d["node"], **{"Raio (mm)": d["radius"], "Espessura corroída": d["t"]})
        refs[-1] = ("corrosion", i)
    if not rows:
        return
    st.markdown("**Condições atribuídas**")
    st.caption("Edite os valores diretamente na tabela; cada alteração recalcula a análise. "
               "Nas arestas, célula vazia = GDL livre; arestas sem nenhum GDL são removidas.")
    b_minus, b_plus, b_cap = st.columns([1, 1, 8], vertical_alignment="center")
    factor = 0.9 if b_minus.button("−", key="sum_minus", help="Reduz 10% as pressões, forças e "
                                   "valores prescritos nas arestas") else \
        1.1 if b_plus.button("+", key="sum_plus", help="Aumenta 10% as pressões, forças e "
                             "valores prescritos nas arestas") else None
    b_cap.caption("− / + : cada clique altera em 10 % pressões, forças e valores prescritos "
                  "(raio e espessura corroída não mudam).")
    if factor:
        for dofs in bc["curves"].values():
            for d in dofs:
                dofs[d] *= factor
        for tag in bc["pressure"]:
            bc["pressure"][tag] *= factor
        for tag, q in bc["edge_forces"].items():
            bc["edge_forces"][tag] = [v * factor for v in q]
        for l in bc["forces"]:
            l["F"] = [v * factor for v in l["F"]]
        st.session_state.recalc = True
        st.rerun()
    edited = st.data_editor(
        pd.DataFrame(rows), hide_index=True, width="stretch", disabled=["Tipo", "ID"],
        key=f"summary-{sig}",
        column_config={c: st.column_config.NumberColumn(format="%.6g") for c in cols})

    changed = False
    invalid = False
    for (kind, ref), (_, row) in zip(refs, edited.iterrows()):
        if kind == "curve":
            new = {d: float(row[d]) for d in DOFS if pd.notna(row[d])}
            if new != bc["curves"][ref]:
                if new:
                    bc["curves"][ref] = new
                else:
                    bc["curves"].pop(ref)
                changed = True
        elif kind == "pressure":
            if pd.notna(row["Pressão"]) and float(row["Pressão"]) != bc["pressure"][ref]:
                bc["pressure"][ref] = float(row["Pressão"])
                changed = True
        elif kind == "edge_force":
            q = [row["FX (N)"], row["FY (N)"], row["FZ (N)"]]
            if all(pd.notna(v) for v in q) and [float(v) for v in q] != bc["edge_forces"][ref]:
                bc["edge_forces"][ref] = [float(v) for v in q]
                changed = True
        elif kind == "force":
            F = [row["FX (N)"], row["FY (N)"], row["FZ (N)"]]
            if all(pd.notna(v) for v in F) and [float(v) for v in F] != bc["forces"][ref]["F"]:
                bc["forces"][ref]["F"] = [float(v) for v in F]
                changed = True
        else:
            r, t = row["Raio (mm)"], row["Espessura corroída"]
            d = bc["corrosion"][ref]
            if pd.isna(r) or pd.isna(t) or r <= 0 or not 0 < t < nominal:
                invalid = invalid or (float(r), float(t)) != (d["radius"], d["t"])
            elif (float(r), float(t)) != (d["radius"], d["t"]):
                d["radius"], d["t"] = float(r), float(t)
                changed = True
    if invalid and not changed:
        st.warning(f"Corrosão: raio deve ser > 0 e espessura corroída entre 0 e {nominal:g} (exclusive).")
    if changed:
        st.session_state.recalc = True
        st.rerun()


FIELDS = {
    "Von Mises (MPa)": "vm",
    "Máxima tensão principal (MPa)": "p1",
    "Mínima tensão principal (MPa)": "p2",
    "Deformação equivalente (%)": "eps",
    "Deslocamento resultante (mm)": "disp",
    "Deslocamento em X (mm) UX": "ux",
    "Deslocamento em Y (mm) UY": "uy",
    "Deslocamento em Z (mm) UZ": "uz",
}
COMPONENTS = {"ux": 0, "uy": 1, "uz": 2}
LAYER_DEFAULT = "PL+PB_top — Tensão de Membrana + Flexão"
VM_LAYERS = {
    "PL — Tensão de Membrana": ("mid", "PL"),
    "PL+PB_bottom — Tensão de Membrana + Flexão": ("bottom", "PL+PB_bottom"),
    "PL+PB_top — Tensão de Membrana + Flexão": ("top", "PL+PB_top"),
}


def fmt_vec(v):
    return ", ".join(f"{x:.4g}" for x in v)


def equilibrium_report(res):
    eq = res.equilibrium
    if eq.ok:
        st.success("Equilíbrio estático verificado: " + eq.messages[0])
    else:
        st.error("Equilíbrio estático NÃO satisfeito - análise não executada. " + " ".join(eq.messages))
    rows = [
        {"Verificação": "Resultante das cargas - força (Fx, Fy, Fz)", "Valor": fmt_vec(eq.force)},
        {"Verificação": "Resultante das cargas - momento (Mx, My, Mz)", "Valor": fmt_vec(eq.moment)},
        {"Verificação": "Movimentos de corpo rígido restringidos", "Valor": f"{eq.restrained_modes} de 6"},
    ]
    if res.solved:
        p = res.post
        rows += [
            {"Verificação": "Reações totais - força", "Valor": fmt_vec(p["reaction_force"])},
            {"Verificação": "Reações totais - momento", "Valor": fmt_vec(p["reaction_moment"])},
            {"Verificação": "Erro de equilíbrio (cargas + reações), força",
             "Valor": f"{p['force_error']:.2e}"},
            {"Verificação": "Erro de equilíbrio (cargas + reações), momento",
             "Valor": f"{p['moment_error']:.2e}"},
        ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


@st.cache_data(show_spinner="Lendo isométrico...")
def read_iso(data: bytes):
    return isometric.parse_isometric(data)


@st.cache_data(show_spinner="Gerando a casca da tubulação...")
def pipe_step(pts: list, mean_diameter: float, radius: float) -> bytes:
    return isometric.build_pipe_step(np.array(pts), mean_diameter, radius)


def iso_input(data: bytes):
    """Isométrico PDF -> bytes STEP da casca; devolve também a espessura de parede."""
    try:
        iso = read_iso(data)
    except Exception as exc:
        st.error(f"Falha ao ler o isométrico: {exc}")
        st.stop()
    return iso_ui(iso, hashlib.md5(data).hexdigest()[:8], "Isométrico (PDF) → geometria da tubulação")


def manual_iso_input():
    iso = isometric.IsoResult(
        segments=[isometric.Segment("E", 1, 3000.0, "manual"), isometric.Segment("U", 1, 2000.0, "manual"),
                  isometric.Segment("N", 1, 3000.0, "manual")],
        nps=6.0, wall=isometric.SCH40[6.0])
    return iso_ui(iso, "manual", "Entrada manual → geometria da tubulação")


def iso_ui(iso, sig: str, title: str):
    with st.expander(title, expanded=True):
        c_img, c_par = st.columns([1.3, 1])
        if iso.page_png:
            c_img.image(iso.page_png, caption="Isométrico lido (linha de centro = traço mais grosso)")
        else:
            c_img.info("Monte o roteamento na tabela abaixo: cada linha é um trecho reto, partindo da "
                       "origem; as curvas entre trechos são geradas automaticamente.")
        with c_par:
            wall = st.number_input("Espessura de parede (mm)", value=float(iso.wall or 7.11),
                                   min_value=0.1, format="%.2f", key=f"wall-{sig}")
            factor = st.number_input("Raio das curvas (× DN)", value=isometric.BEND_FACTOR,
                                     min_value=1.0, step=0.5, key=f"bf-{sig}")
            cond = st.segmented_control(
                "Pressão lida do isométrico", ["Projeto", "Operação"], default="Projeto",
                key=f"pcond-{sig}", disabled=iso.pressure_design is None)
            p_read = iso.pressure_op if cond == "Operação" else iso.pressure_design
            p_int = st.number_input("Pressão interna (MPa)", value=float(p_read or 0.0),
                                    min_value=0.0, format="%.6g", key=f"pint-{sig}-{cond}",
                                    help="Lida da tabela do isométrico (bar g → MPa) e aplicada "
                                         "automaticamente como pressão interna em todas as superfícies.")
            t_read = iso.temp_op if cond == "Operação" else iso.temp_design
            t_int = t_read if t_read is not None else None
        dirs = {f"{'+' if sg > 0 else '−'}{lab}": (ax, sg)
                for ax, lab in (("E", "X"), ("N", "Y"), ("U", "Z")) for sg in (1, -1)}
        label_of = {v: k for k, v in dirs.items()}
        table = pd.DataFrame({
            "Direção": [label_of.get((s.axis, s.sign), "+Y") for s in iso.segments],
            "Comprimento (mm)": [s.length for s in iso.segments],
            "Origem": [s.source for s in iso.segments],
        })
        st.markdown("**Trechos do isométrico** — edite comprimentos e direções; use **+** na tabela "
                    "para inserir uma linha e selecione-a (caixa à esquerda) + *Delete* para remover.")
        n_par, n_tab = st.columns([1, 2.2])
        with n_par:
            nps = st.number_input("DN (pol)", value=float(iso.nps or 6.0), min_value=0.5, step=0.5,
                                  key=f"nps-{sig}")
            od_default = isometric.OD_TABLE.get(nps, nps * isometric.IN)
            od = st.number_input("Diâmetro externo (mm)", value=float(od_default), min_value=1.0,
                                 format="%.2f", key=f"od-{sig}-{nps}")
        with n_tab:
            edited = st.data_editor(
                table, hide_index=True, use_container_width=True, key=f"seg-{sig}",
                num_rows="dynamic",
                column_config={
                    "Direção": st.column_config.SelectboxColumn(options=list(dirs), required=True,
                                                                default="+Y"),
                    "Comprimento (mm)": st.column_config.NumberColumn(
                        min_value=1.0, format="%.1f", required=True, default=1000.0),
                    "Origem": st.column_config.TextColumn(default="manual")})
        radius = factor * nps * isometric.IN
        st.caption(f"Raio de curvatura = {factor:g} × {nps:g}\" = {radius / isometric.IN:.2f}\" "
                   f"({radius:.1f} mm). A casca fica na superfície média (Ø = DE − e).")
        for m in iso.messages:
            st.warning(m)
    edited = edited.dropna(subset=["Direção", "Comprimento (mm)"])
    if edited.empty:
        st.error("Inclua ao menos um trecho na tabela.")
        st.stop()
    segs = [isometric.Segment(*dirs[d], float(L), str(o))
            for d, L, o in zip(edited["Direção"], edited["Comprimento (mm)"], edited["Origem"])]
    pts = isometric.centerline_points(segs)
    st.session_state.report_geom = {
        "info": [("DN", f"{nps:g} pol"), ("Diâmetro externo", f"{od:g} mm"),
                 ("Espessura de parede", f"{wall:g} mm"), ("Raio das curvas", f"{factor:g} × DN = {radius:.1f} mm"),
                 ("Pressão interna", f"{p_int:g} MPa"),
                 ("Temperatura", "n/d" if t_int is None else f"{t_int:g} °C")],
        "tables": [("Trechos do roteamento (a partir da origem)", ["Trecho", "Direção", "Comprimento (mm)"],
                    [[str(i), d, f"{L:g}"] for i, (d, L) in
                     enumerate(zip(edited["Direção"], edited["Comprimento (mm)"]), 1)])]}
    try:
        step = pipe_step(pts.round(4).tolist(), float(od - wall), float(radius))
    except Exception as exc:
        st.error(f"Não foi possível gerar a tubulação: {exc}")
        st.stop()
    return step, float(wall), float(od), float(p_int), t_int


@st.cache_data(show_spinner="Gerando o vaso de pressão...")
def vessel_step(params: vessel.VesselParams) -> bytes:
    return vessel.build_vessel_step(params)


def vessel_input():
    """Tabela de dimensões -> STEP do vaso; devolve (step, params, espessura, DE, pressão)."""
    with st.expander("Entrada manual → vaso de pressão", expanded=True):
        st.info("Vaso vertical: costado cilíndrico com a base solta (aresta livre, ancorada), tampo "
                "semi-elíptico 2:1 no topo, um bocal no topo (sobre o tampo) e um bocal no costado "
                "(direção +Y). A casca fica na superfície média. Unidades: mm.")
        table = pd.DataFrame({
            "Componente": ["Costado", "Tampo 2:1", "Bocal do topo", "Bocal do costado"],
            "Diâmetro externo (mm)": [1000.0, None, 200.0, 150.0],
            "Espessura (mm)": [10.0, 12.0, 8.0, 6.0],
            "Comprimento (mm)": [2000.0, None, 300.0, 250.0],
        })
        edited = st.data_editor(
            table, hide_index=True, use_container_width=True, key="vessel-table",
            disabled=["Componente"],
            column_config={c: st.column_config.NumberColumn(min_value=0.0, format="%.2f")
                           for c in table.columns[1:]})
        st.caption("Costado: comprimento reto, da base à linha de tangência. Tampo: mesmo diâmetro do "
                   "costado (profundidade = D/4). Bocais: comprimento projetado a partir da superfície "
                   "do vaso (topo: acima do ápice do tampo; costado: a partir da parede).")
        c1, c2 = st.columns(2)
        c1.caption("Bocal do costado no meio da altura do costado. Tampas dos bocais: 50 mm.")
        p_int = c2.number_input("Pressão interna (MPa)", value=0.0, min_value=0.0, format="%.6g",
                                key="vessel-p", help="Aplicada como pressão interna em todas as superfícies.")
        st.markdown("**Chapas de reforço (pad)** — disco com o dobro do raio do bocal, centrado nele; "
                    "a espessura informada é a total na região do reforço (padrão: 2× a do tampo/costado).")
        ev = edited.set_index("Componente")["Espessura (mm)"]
        head_t0 = float(ev["Tampo 2:1"]) if pd.notna(ev["Tampo 2:1"]) else 10.0
        shell_t0 = float(ev["Costado"]) if pd.notna(ev["Costado"]) else 10.0
        r1, r2 = st.columns(2)
        with r1:
            top_pad = st.checkbox("Reforço no bocal do topo", value=False, key="pad-top")
            top_pad_t = st.number_input(
                "Espessura do reforço do topo (mm)", value=2 * head_t0, min_value=0.1, format="%.2f",
                disabled=not top_pad, key=f"pad-top-t-{head_t0:g}")
        with r2:
            side_pad = st.checkbox("Reforço no bocal do costado", value=False, key="pad-side")
            side_pad_t = st.number_input(
                "Espessura do reforço do costado (mm)", value=2 * shell_t0, min_value=0.1, format="%.2f",
                disabled=not side_pad, key=f"pad-side-t-{shell_t0:g}")
    v = edited.set_index("Componente")
    try:
        g = lambda row, col: float(v.loc[row, col])
        params = vessel.VesselParams(
            shell_od=g("Costado", "Diâmetro externo (mm)"), shell_t=g("Costado", "Espessura (mm)"),
            shell_len=g("Costado", "Comprimento (mm)"), head_t=g("Tampo 2:1", "Espessura (mm)"),
            top_od=g("Bocal do topo", "Diâmetro externo (mm)"), top_t=g("Bocal do topo", "Espessura (mm)"),
            top_len=g("Bocal do topo", "Comprimento (mm)"),
            side_od=g("Bocal do costado", "Diâmetro externo (mm)"),
            side_t=g("Bocal do costado", "Espessura (mm)"),
            side_len=g("Bocal do costado", "Comprimento (mm)"),
            top_pad_t=float(top_pad_t) if top_pad else 0.0,
            side_pad_t=float(side_pad_t) if side_pad else 0.0)
        if any(np.isnan(x) for x in params.__dict__.values()):
            raise ValueError("Preencha todos os diâmetros, espessuras e comprimentos.")
        step = vessel_step(params)
        st.session_state.report_geom = {
            "info": [("Configuração", "Vaso vertical: costado cilíndrico (base solta), tampo semi-elíptico 2:1 "
                                      "no topo, bocal no topo e bocal no costado (a meia altura)"),
                     ("Pressão interna", f"{p_int:g} MPa"),
                     ("Tampas planas dos bocais", f"{vessel.CAP_T:g} mm"),
                     ("Reforço do bocal do topo", f"{params.top_pad_t:g} mm (raio {params.top_pad_r:.1f} mm)"
                      if params.top_pad_t else "sem reforço"),
                     ("Reforço do bocal do costado", f"{params.side_pad_t:g} mm (raio {params.side_pad_r:.1f} mm)"
                      if params.side_pad_t else "sem reforço")],
            "tables": [("Dimensões dos componentes", ["Componente", "DE (mm)", "Espessura (mm)", "Comprimento (mm)"],
                        [[c, "—" if pd.isna(a) else f"{a:g}", f"{t:g}", "—" if pd.isna(l) else f"{l:g}"]
                         for c, a, t, l in edited[["Componente", "Diâmetro externo (mm)", "Espessura (mm)",
                                                   "Comprimento (mm)"]].itertuples(index=False)])]}
    except Exception as exc:
        st.error(f"Não foi possível gerar o vaso: {exc}")
        st.stop()
    return step, params, p_int


def manual_start():
    """Tela inicial sem arquivo: escolha da entrada manual. Retorna o tipo escolhido ou None."""
    if "manual" not in st.session_state:
        st.session_state.manual = None
    if st.session_state.manual is None:
        st.info("Envie um arquivo .step/.stp/.stl ou um isométrico .pdf na barra lateral, ou crie a "
                "geometria manualmente.")
        if st.button("Entrada manual da geometria", type="primary"):
            st.session_state.manual = "choose"
            st.rerun()
        st.stop()
    c1, c2 = st.columns([4, 1])
    kind = c1.radio("Geometria a criar", ["Tubulação (isométrico)", "Vaso de pressão"],
                    horizontal=True, key="manual_kind")
    if c2.button("Voltar"):
        st.session_state.manual = None
        st.rerun()
    return "pipe" if kind.startswith("Tub") else "vessel"


if uploaded is None:
    manual_kind = manual_start()
else:
    manual_kind = None

raw = uploaded.getvalue() if uploaded is not None else b""
suffix = os.path.splitext(uploaded.name)[1] if uploaded is not None else ".step"
iso_wall = iso_od = iso_pressure = iso_temp = None
vessel_params = None
st.session_state.report_geom = None
if is_pdf:
    raw, iso_wall, iso_od, iso_pressure, iso_temp = iso_input(raw)
    suffix = ".step"
elif manual_kind == "pipe":
    raw, iso_wall, iso_od, iso_pressure, iso_temp = manual_iso_input()
elif manual_kind == "vessel":
    raw, vessel_params, iso_pressure = vessel_input()
    iso_wall, iso_od = vessel_params.shell_t, vessel_params.shell_od
file_sig = hashlib.md5(raw).hexdigest()
try:
    mesh = build(raw, suffix, divisions, refinements)
except Exception as exc:
    st.error(f"Falha ao gerar a malha: {exc}")
    st.stop()
if vessel_params is not None:
    mesh = replace(mesh, base_thickness=vessel.element_thickness(mesh.nodes, mesh.triangles, vessel_params))

if len_scale != 1.0:
    mesh = replace(
        mesh, nodes=mesh.nodes * len_scale, bbox_diagonal=mesh.bbox_diagonal * len_scale,
        curve_length={t: l * len_scale for t, l in mesh.curve_length.items()})

if st.session_state.get("file_sig") != file_sig:
    st.session_state.file_sig = file_sig
    st.session_state.bc = {"curves": {}, "pressure": {}, "corrosion": [], "forces": [],
                           "edge_forces": {}, "cracks": []}
    st.session_state.sel = {"curves": [], "surfaces": [], "node": None}
    # Arestas soltas (livres) são ancoradas nos 6 GDL e a análise roda logo ao subir a geometria.
    free_edges = free_curves(mesh)
    for tag in free_edges:
        st.session_state.bc["curves"][tag] = {d: 0.0 for d in DOFS}
    if free_edges:
        # Sem isométrico não há pressão conhecida: espera o usuário informar a pressão
        if iso_pressure:
            st.session_state.recalc = True
        st.toast(f"Arestas soltas ancoradas (6 GDL = 0): {', '.join(map(str, free_edges))}")
bc = st.session_state.bc
bc.setdefault("corrosion", [])
bc.setdefault("forces", [])
bc.setdefault("edge_forces", {})
if not isinstance(bc.get("cracks"), list):
    bc["cracks"] = []
# Node indices change with the mesh, so corrosion defects are tied to it.
mesh_key = (file_sig, divisions, refinements)
if st.session_state.get("mesh_key") != mesh_key:
    st.session_state.mesh_key = mesh_key
    if bc["corrosion"] or bc["forces"]:
        bc["corrosion"].clear()
        bc["forces"].clear()
        st.toast("Malha alterada: os defeitos de corrosão e as forças nodais foram removidos.")
    st.session_state.sel["node"] = None
surface_tags = set(mesh.tri_surface.tolist())
# Pressão interna do isométrico: aplicada (negativa) em todas as superfícies quando lida/alterada.
if iso_pressure:
    p_key = (file_sig, iso_pressure)
    if st.session_state.get("iso_p_applied") != p_key:
        st.session_state.iso_p_applied = p_key
        for tag in surface_tags:
            bc["pressure"][tag] = -iso_pressure


@st.cache_data(show_spinner=False)
def zone_tris(_mesh, key, node: int, radius: float):
    return np.flatnonzero(fea.corrosion_zone(_mesh.nodes, _mesh.triangles, node, radius))

analysis_sig = hashlib.md5(json.dumps(
    [file_sig, divisions, refinements, len_scale, {k: v for k, v in bc.items() if k != "cracks"}], sort_keys=True, default=str).encode()).hexdigest()

def reset_all():
    st.session_state.bc = {"curves": {}, "pressure": {}, "corrosion": [], "forces": [],
                           "edge_forces": {}, "cracks": []}
    st.session_state.sel = {"curves": [], "surfaces": [], "node": None}
    st.session_state.analysis = None
    # New key remounts the viewer, discarding its stored selection.
    st.session_state.reset_n = st.session_state.get("reset_n", 0) + 1


with st.sidebar:
    st.divider()
    st.subheader("Análise")
    E = st.number_input("Módulo de Young E", value=210000.0, min_value=1e-9, format="%.6g")
    nu = st.number_input("Coeficiente de Poisson", value=0.3, min_value=0.0, max_value=0.499)
    thick = st.number_input("Espessura da casca", value=iso_wall or 10.0, min_value=1e-9, format="%.6g",
                            disabled=vessel_params is not None)
    if vessel_params is not None:
        st.caption("As espessuras vêm da tabela de entrada do vaso (costado, tampo e bocais); aqui "
                   "vale a do costado.")
    st.caption("Use unidades consistentes (ex.: mm, N, MPa).")
    use_gravity = st.checkbox(
        "Considerar gravidade (direção −Z)", value=True,
        help="Peso próprio da casca como carga distribuída nos nós, atuando em −Z.")
    gravity = None
    if use_gravity:
        g = st.number_input("Aceleração da gravidade", value=9810.0, min_value=1e-9, format="%.6g",
                            help="9810 mm/s² para mm, N, MPa, t (tonelada).")
        rho_s = st.number_input("Densidade do aço", value=7.85e-9, min_value=1e-15, format="%.4g",
                                help="7,85e-9 t/mm³ (= 7850 kg/m³) com mm, N, MPa.")
        gravity = {"g": g, "rho_steel": rho_s}
        fill = st.checkbox("Tubulação cheia de água", value=True,
                           help="Soma o peso da água ao peso próprio (casca circular).")
        if fill:
            rho_w = st.number_input("Densidade da água", value=1.0e-9, min_value=1e-15,
                                    format="%.4g", help="1,0e-9 t/mm³ (= 1000 kg/m³).")
            d_in = st.number_input(
                "Diâmetro interno (mm)",
                value=float(iso_od - 2 * iso_wall) if iso_od else 100.0,
                min_value=1.0, format="%.2f", key=f"d_in-{iso_od}")
            gravity["water"] = {"rho": rho_w, "ri": d_in / 2}
    # A temperatura define Sp (corrosão) e, se marcado, a dilatação térmica.
    temp_input = st.number_input(
        "Temperatura do metal (°C)", value=float(iso_temp if iso_temp is not None else 20.0),
        min_value=-200.0, max_value=800.0, step=10.0, format="%.6g", key=f"temp_an-{iso_temp}",
        on_change=lambda: st.session_state.update(recalc=True),
        help="Uniforme em todo o modelo. Preenchida com a temperatura lida do isométrico. "
             "Os botões +/− somam 10 °C e qualquer alteração recalcula a análise.")
    S_allow, Sy_yield, Sp = fea.allowable_sp(temp_input)
    if Sp is not None:
        st.caption(f"S = {S_allow:.1f} MPa; Sy = {Sy_yield:.1f} MPa → "
                   f"**Sp = máx(1,5·S; Sy) = {Sp:.1f} MPa**")
    else:
        st.warning("S/Sy não definidos para esta temperatura (tabela de S vai até 475 °C e a de Sy "
                   "até 525 °C): Sp indisponível e a porcentagem de corrosão não é calculada.")
    if iso_od:
        p_applied = max([-v for v in bc["pressure"].values()] + [0.0])
        if p_applied > 0:
            hoop = p_applied * iso_od / (2 * thick)
            msg = f"Hoop = P·D/(2·t) = {p_applied:.3g}·{iso_od:.4g}/(2·{thick:.4g}) = {hoop:.1f} MPa"
            if S_allow is None:
                st.caption(msg)
            elif hoop > S_allow:
                st.caption(msg + f" > S = {S_allow:.1f} MPa")
                st.markdown(":red[**Reprovado por PM (membrana geral) para pressão interna pelo "
                            "ASME Sec VIII Div I**]")
            else:
                st.caption(msg + f" ≤ S = {S_allow:.1f} MPa")
    use_thermal = st.checkbox(
        "Considerar dilatação térmica (aço carbono)", value=True,
        help="Deformação térmica livre = α(T)·(T − 20 °C), com α médio do aço carbono em função "
             "da temperatura (ASME B31.3, Tab. C-1). Referência sem tensão a 20 °C.")
    temperature = None
    if use_thermal:
        temperature = temp_input
        st.caption(f"α médio (20 °C → {temperature:g} °C) = {fea.steel_alpha(temperature) * 1e6:.2f}e-6 /°C; "
                   f"deformação livre = {fea.thermal_strain(temperature) * 100:.4f}%")
    ignore_help = ("Elementos e nós a essa distância (ou menos) de qualquer aresta com restrição ficam "
                   "com tensão/deformação zero e fora dos máximos e das porcentagens de corrosão "
                   "(picos locais de apoio). 0 desativa.")
    default_ign = 300.0 if vessel_params is not None else (float(1.5 * iso_od) if iso_od else 40.0)
    ignore_dist = st.number_input(
        "Desprezar tensões a menos de (mm) das arestas restritas",
        value=default_ign, min_value=0.0, format="%.6g",
        key=f"ignore_dist-{'vessel' if vessel_params is not None else iso_od}", help=ignore_help)
    run = st.button("Verificar equilíbrio e calcular", type="primary")
    st.button("Recomeçar", on_click=reset_all,
              help="Remove todas as cargas, condições de contorno e defeitos de corrosão.")

mat = fea.Material(E, nu, thick)
full_sig = f"{analysis_sig}-{E}-{nu}-{thick}-{json.dumps(gravity, sort_keys=True)}-{temperature}-{ignore_dist}"
if run or st.session_state.pop("recalc", False):
    try:
        with st.spinner("Verificando equilíbrio e calculando..."):
            st.session_state.analysis = {"sig": full_sig,
                                         "result": fea.run_analysis(mesh, mat, bc, gravity, temperature, ignore_dist)}
    except ValueError as exc:
        st.session_state.analysis = None
        st.sidebar.error(str(exc))

analysis = st.session_state.get("analysis")
res = analysis["result"] if analysis and analysis["sig"] == full_sig else None
if analysis and res is None:
    st.sidebar.warning("Dados alterados desde o último cálculo: recalcule para ver os resultados.")

view_nodes, view_ghost, view_field, view_label = None, None, None, ""
view_dir = None
view_dir2 = None
gen_report = False
view_sgn = view_sgn2 = None
view_dir_min = False
view_minimum = False
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nós", len(mesh.nodes))
c2.metric("Elementos (triângulos)", len(mesh.triangles))
c3.metric("Superfícies", len(surface_tags))
c4.metric("Arestas", len(mesh.curves))
if mesh.welded:
    st.caption(f"Pré-processamento: {mesh.welded} nós duplicados nas arestas comuns foram unidos.")
left, right = st.columns([3, 1.4])
with right:
    sel_mode = st.radio("Seleção no modelo",
                        ["Arestas e superfícies", "Nó (corrosão)", "Nó (força)"], key="sel_mode")
node_mode = sel_mode.startswith("Nó")
corr_mode = sel_mode == "Nó (corrosão)"
if res is not None and res.solved:
    auto = 0.05 * mesh.bbox_diagonal / res.u_max if res.u_max > 0 else 1.0
    with st.sidebar:
        st.divider()
        st.subheader("Resultados")
        show_res = st.checkbox("Mostrar deformada e resultados no modelo", value=True)
        with left:
            field_name = st.selectbox("Campo", list(FIELDS))
        layer = None
        if FIELDS[field_name] in ("vm", "eps", "p1", "p2"):
            with left:
                layer = st.selectbox("Posição na espessura", list(VM_LAYERS),
                                     index=list(VM_LAYERS).index(LAYER_DEFAULT))
        # Os controles saem da tela quando não há resultado válido e o Streamlit descarta o estado
        # deles; por isso o valor travado é guardado em chaves próprias e restaurado aqui.
        if "lock_scale" not in st.session_state:
            st.session_state.lock_scale = st.session_state.get("lock_scale_saved", False)
        lock_scale = st.checkbox(
            "Travar fator de escala da deformação", key="lock_scale",
            help="Mantém o fator atual nas próximas análises, em vez de voltar ao valor automático.")
        # O valor inicial da barra é passado explicitamente (sem key): quando o limite ou o passo
        # mudam entre análises o Streamlit recria a barra, e sem isso ela voltaria a 0.
        saved = st.session_state.get("def_scale_saved")
        init = float(saved) if lock_scale and saved is not None else float(auto)
        scale = st.slider(
            "Fator de escala da deformação", 0.0, float(max(auto * 10, init)), init,
            step=float(auto / 50), format="%.4g",
            help="Multiplica os deslocamentos apenas na visualização (1 = real). "
                 f"O valor inicial ({auto:.4g}) deixa o deslocamento máximo em 5% da "
                 "diagonal do modelo.")
        st.session_state.lock_scale_saved = lock_scale
        st.session_state.def_scale_saved = float(scale)
        show_ghost = st.checkbox("Mostrar malha indeformada (referência)", value=False)
        gen_report = st.button("Gerar relatório PDF", key="gen_report",
                               help="PDF com entradas, saídas, code compliance e mapas 3D de Von Mises "
                                    "(meio, topo e base da espessura).")
    if show_res:
        view_minimum = FIELDS[field_name] == "p2"
        if FIELDS[field_name] == "disp":
            view_label = "Deslocamento resultante |U| (mm)"
            view_field = np.linalg.norm(res.u[:, :3], axis=1)
        elif FIELDS[field_name] in COMPONENTS:
            view_label = f"Deslocamento {field_name[-2:]} (mm)"
            view_field = res.u[:, COMPONENTS[FIELDS[field_name]]]
        else:
            suffix, view_label = VM_LAYERS[layer]
            prefix, unit = {"eps": ("eps_", "Deformação equivalente (%)"),
                            "p1": ("p1_", "Máxima tensão principal (MPa)"),
                            "p2": ("p2_", "Mínima tensão principal (MPa)")}.get(
                FIELDS[field_name], ("", "Von Mises (MPa)"))
            view_label = f"{unit} — {view_label}"
            view_field = res.vm_nodal[prefix + suffix]
            if FIELDS[field_name] in ("p1", "p2"):
                s1 = np.abs(res.vm_nodal["p1_" + suffix])
                s2 = np.abs(res.vm_nodal["p2_" + suffix])
                big = np.maximum(np.maximum(s1, s2), 1e-300)
                dir1 = res.vm_nodal["d1_" + suffix] * (s1 / big)[:, None]
                dir2 = res.vm_nodal["d2_" + suffix] * (s2 / big)[:, None]
                sg1 = np.sign(res.vm_nodal["p1_" + suffix])
                sg2 = np.sign(res.vm_nodal["p2_" + suffix])
                view_dir = dir1 if FIELDS[field_name] == "p1" else dir2
                view_dir2 = dir2 if FIELDS[field_name] == "p1" else None
                view_sgn = sg1 if FIELDS[field_name] == "p1" else sg2
                view_sgn2 = sg2 if FIELDS[field_name] == "p1" else None
                view_dir_min = FIELDS[field_name] == "p2"
        view_nodes = mesh.nodes + res.u[:, :3] * scale
        if show_ghost and scale > 0:
            view_ghost = mesh.nodes
        st.sidebar.caption(f"Deformada ampliada {scale:.4g}× — deslocamento máx. real "
                           f"{res.u_max:.4g}, visualizado {res.u_max * scale:.4g}.")

with left:
    zone_key = (file_sig, divisions, refinements, len_scale)
    corr_tris = [zone_tris(mesh, zone_key, d["node"], d["radius"]) for d in bc["corrosion"]]
    sel_node = st.session_state.sel.get("node")
    if corr_mode and sel_node is not None:
        corr_tris.append(zone_tris(mesh, zone_key, sel_node,
                                   float(st.session_state.get("corr_radius", 100.0))))
    # Na tela de resultados só as tensões são plotadas, sem destacar a perda de espessura
    corr_union = np.unique(np.concatenate(corr_tris)) if corr_tris and view_field is None else []
    corr_flags = []
    if view_field is None:
        corr_flags = [{"node": int(d["node"]), "text": f"LTA {i}"}
                      for i, d in enumerate(bc["corrosion"], 1)]
    elif S_allow is not None and Sp is not None:
        # Admissível: S no meio da espessura (PL); máx(1,5·S; Sy) = Sp no topo/base (PL+PB).
        # A camada segue a posição escolhida quando há campo de tensão; senão usa o meio.
        c_layer = VM_LAYERS[layer][0] if layer else "mid"
        c_allow = S_allow if c_layer == "mid" else Sp
        c_vm = res.vm_element[c_layer]
        for i, (d, zone) in enumerate(zip(bc["corrosion"], corr_tris), 1):
            if len(zone) == 0:
                continue
            k = zone[np.argmax(c_vm[zone])]
            tri = mesh.triangles[k]
            ratio = c_vm[k] / c_allow * 100
            verdict = "Reprovado" if ratio > 100 else "Aprovado"
            corr_flags.append({"node": int(tri[np.argmax(res.vm_nodal[c_layer][tri])]),
                               "text": f"LTA {i} - {ratio:.0f}% - {verdict} - ASME API-579 part 5 Nível 3"})
    elif view_field is not None:
        corr_flags = [{"node": int(d["node"]), "text": f"LTA {i}"}
                      for i, d in enumerate(bc["corrosion"], 1)]
    max_note = ""
    if view_field is not None and FIELDS[field_name] == "vm" and Sp is not None:
        # Utilização no nó de pico do campo exibido (PL, PL+PB_bottom ou PL+PB_top)
        ratio_max = float(np.max(view_field)) / Sp * 100
        verdict_max = "Reprovado" if ratio_max > 100 else "Aprovado"
        max_note = f"{ratio_max:.0f}% - {verdict_max} - ASME Sec VIII Div 2 Part 5"
    result = mesh_viewer(
        mesh, f"{file_sig}-{divisions}-{refinements}-{len_scale}", st.session_state.sel,
        {t: "anchor" if is_anchor(d) else "custom" for t, d in bc["curves"].items()},
        bc["pressure"], show_mesh, nodes=view_nodes, ghost=view_ghost, field=view_field, field_label=view_label,
        mode="node" if node_mode else "edge", corr_tris=corr_union, corr_flags=corr_flags if show_flags else [],
        max_note=max_note, field_minimum=view_minimum,
        field_dir=view_dir, field_dir2=view_dir2, field_sgn=view_sgn, field_sgn2=view_sgn2,
        field_dir_min=view_dir_min,
        crack_edges=sorted({c["edge"] for c in bc["cracks"]}),
        crack_flags=[{"node": int(mesh.curves[c["edge"]][len(mesh.curves[c["edge"]]) // 2][0]),
                      "text": f"Trinca {c['id']} - Aprovada - API-579 Part 9 Nivel 2, MAT = -26°C"} for c in bc["cracks"] if show_flags and c["edge"] in mesh.curves],
        key=f"viewer-{st.session_state.get('reset_n', 0)}",
    )
    picked = result.selection if result is not None else None
    if picked:
        node = picked.get("node")
        st.session_state.sel = {
            "curves": [c for c in picked.get("curves", []) if c in mesh.curves],
            "surfaces": [s for s in picked.get("surfaces", []) if s in surface_tags],
            "node": node if isinstance(node, int) and 0 <= node < len(mesh.nodes) else None,
        }
sel = st.session_state.sel

with right:
    if corr_mode:
        with st.container(border=True):
            st.subheader("Corrosão (nó)")
            corrosion_form(sel.get("node"), bc, mesh, thick)
    elif node_mode:
        with st.container(border=True):
            st.subheader("Força (nó)")
            force_form(sel.get("node"), bc, mesh)
    else:
        if not sel["curves"] and not sel["surfaces"]:
            st.info("Selecione arestas ou superfícies no modelo para atribuir condições.")
        if sel["curves"]:
            with st.container(border=True):
                st.subheader("Restrições e cargas (aresta)")
                edge_form(sel["curves"], bc, mesh.curve_length)
                st.divider()
                edge_load_form(sel["curves"], bc, mesh.curve_length)
                st.divider()
                crack_form(sel["curves"], bc, mesh, res)
        if sel["surfaces"]:
            with st.container(border=True):
                st.subheader("Pressão (superfície)")
                surface_form(sel["surfaces"], bc, sorted(surface_tags))
with right:
    if res is not None and res.solved:
        st.markdown("#### Code Compliance - ASME Sec VIII Div 2 Part 5")
        p_in = max([-v for v in bc["pressure"].values()] + [0.0])
        hoop = p_in * iso_od / (2 * thick) if iso_od and p_in > 0 else None
    
        def status(act, allow):
            if act is None or allow is None:
                return "n/d"
            return ":red[**REPROVADO**]" if act > allow else "APROVADO"
    
        def cell(v):
            return "n/d" if v is None else f"{v:.1f}"
    
        pl = float(res.vm_nodal["mid"].max())
        pb = float(max(res.vm_nodal["top"].max(), res.vm_nodal["bottom"].max()))
        rows = [
            ("P_M", "Membrana geral", S_allow, hoop),
            ("P_L", "Membrana local", Sp, pl),
            ("P_L + P_B", "Membrana Local + Flexão", Sp, pb),
        ]
        sub = {"P_M": "P<sub>M</sub>", "P_L": "P<sub>L</sub>", "P_L + P_B": "P<sub>L</sub> + P<sub>B</sub>"}
        compliance_rows = [(sub[c], d, a, v, "n/d" if v is None or a is None else
                            ("REPROVADO" if v > a else "APROVADO")) for c, d, a, v in rows]
        lines = ["| Categoria de tensões | Descrição | Admissível (MPa) | Atuante (MPa) | Status |",
                 "|---|---|---|---|---|"]
        for cat, desc, allow, act in rows:
            lines.append(f"| {sub[cat]} | {desc} | {cell(allow)} | {cell(act)} | {status(act, allow)} |")
        st.markdown("\n".join(lines), unsafe_allow_html=True)
        st.caption("P_M: admissível S, atuante hoop = P·D/(2t) (precisa de isométrico e pressão interna). "
                   "P_L: admissível máx(1,5·S; Sy), atuante Von Mises máx. no meio da espessura. "
                   "P_L+P_B: admissível máx(1,5·S; Sy), atuante Von Mises máx. no topo ou base. "
                   "Os atuantes usam o valor nodal suavizado, fora da zona desprezada junto às restrições.")
        if bc["cracks"]:
            st.markdown("#### Tensões normais à trinca (que abrem a trinca) no cordão de solda selecionado")
            crack_rows = []
            for c in bc["cracks"]:
                if c.get("stress") is None:
                    c["stress"] = crack_stress(mesh, res, c["edge"])
                v = c["stress"]
                crack_rows.append({
                    "Trinca": f"Trinca {c['id']}",
                    "Aresta": c["edge"],
                    "Dimensões": crack_text(c),
                    "PL (MPa)": None if v is None else v["mid"],
                    "PL+PB_top (MPa)": None if v is None else v["top"],
                    "PL+PB_bottom (MPa)": None if v is None else v["bottom"],
                })
            st.dataframe(pd.DataFrame(crack_rows), hide_index=True, width="stretch",
                         column_config={k: st.column_config.NumberColumn(format="%.1f")
                                        for k in ("PL (MPa)", "PL+PB_top (MPa)", "PL+PB_bottom (MPa)")})
            st.caption("Valores salvos no momento em que a trinca foi adicionada (se ainda não havia "
                       "resultado, no primeiro resultado disponível). Maior tensão normal (com sinal, "
                       "tração positiva) no plano da casca e ortogonal à aresta, entre os elementos "
                       "adjacentes a ela; PL = meio da espessura, PL+PB_top/bottom = superfícies. "
                       "Elementos na zona desprezada junto às restrições são ignorados.")
with left:
    summary(bc, thick, analysis_sig)

if res is not None:
    st.divider()
    st.subheader("Resultados")
    equilibrium_report(res)
    if res.solved:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Von Mises máx. (elemento)", f"{res.vm_element['max'].max():.4g}")
        m2.metric("Von Mises máx. (nodal suavizado)", f"{res.vm_nodal['max'].max():.4g}")
        m3.metric("Deslocamento máx.", f"{res.u_max:.4g}")
        m4.metric(f"Sp = máx(1,5·S; Sy) a {temp_input:g} °C (MPa)",
                  f"{Sp:.1f}" if Sp is not None else "indisponível",
                  help="S e Sy interpolados das tabelas por temperatura; Sp é a referência da "
                       "porcentagem das áreas corroídas.")
        st.caption("Tensões em elemento constante (CST) na superfície topo/base da casca; o campo "
                   "mostrado é a média nodal ponderada por área. Pico de tensão em apoios/quinas "
                   "depende da malha: refine para verificar convergência.")

    if res.solved:
        report_sig = f"{full_sig}-{scale}-{uploaded.name if uploaded else manual_kind}"
        if gen_report:
            crack_tbl = []
            for c in bc["cracks"]:
                if c.get("stress") is None:
                    c["stress"] = crack_stress(mesh, res, c["edge"])
                v = c["stress"]
                crack_tbl.append([f"Trinca {c['id']}", str(c["edge"]), crack_text(c)] +
                                 ["n/d" if v is None else f"{v[k]:.1f}" for k in ("mid", "top", "bottom")])
            inputs = [("Módulo de Young E", f"{E:g} MPa"), ("Coeficiente de Poisson", f"{nu:g}"),
                      ("Espessura de referência (campo lateral)", f"{thick:g} mm"),
                      ("Gravidade", "não considerada" if not gravity else
                       f"g = {gravity['g']:g} mm/s², aço ρ = {gravity['rho_steel']:.3g} t/mm³ (−Z)"),
                      ("Enchimento com água", "não" if not gravity or "water" not in gravity else
                       f"ρ = {gravity['water']['rho']:.3g} t/mm³, diâmetro interno "
                       f"{2 * gravity['water']['ri']:g} mm"),
                      ("Dilatação térmica", "não considerada" if temperature is None else
                       f"α médio = {fea.steel_alpha(temperature) * 1e6:.2f}e-6 /°C "
                       f"(20 °C → {temperature:g} °C)"),
                      ("Zona desprezada junto às restrições", f"{ignore_dist:g} mm")]
            notes = [
                "Modelo de casca em superfície média com elementos triangulares de deformação constante (CST) "
                "e flexão de placa; tensões de membrana e de flexão avaliadas no meio, topo e base da espessura.",
                "A convergência de malha deve ser verificada pelo engenheiro (refinar e comparar picos de tensão).",
                "Tensões nodais são médias ponderadas por área; picos locais em apoios e quinas são sensíveis à "
                "malha e foram desprezados na zona informada junto às arestas restritas.",
                "O código de aceitação aplicado é o ASME Sec. VIII Div. 2 Part 5 (categorias PM, PL e PL+PB) e as "
                "tabelas S/Sy interpoladas por temperatura; confira com a edição vigente da norma.",
                "Resultados válidos somente para os dados de entrada e as condições de contorno listados neste "
                "relatório. Soldas, juntas, cargas de vento/sísmicas e fadiga não foram consideradas, salvo se "
                "listadas nas entradas.",
            ]
            geometry = st.session_state.get("report_geom") or {"info": [], "tables": []}
            if uploaded is not None and not is_pdf:
                geometry = {"info": [("Arquivo", uploaded.name)], "tables": []}
            with st.spinner("Gerando relatório PDF (renderizando figuras 3D)..."):
                pdf = report.generate({
                    "mesh": mesh, "res": res, "bc": bc, "scale": float(scale),
                    "S_allow": S_allow, "Sy": Sy_yield, "Sp": Sp, "temp": temp_input, "thick": thick,
                    "source": uploaded.name if uploaded is not None else
                    ("Entrada manual — vaso de pressão" if manual_kind == "vessel"
                     else "Entrada manual — tubulação"),
                    "kind": "Vaso de pressão" if vessel_params is not None else
                    ("Tubulação (isométrico)" if is_pdf or manual_kind == "pipe" else "Geometria importada (casca)"),
                    "geometry": geometry, "divisions": divisions, "refinements": refinements,
                    "len_scale": len_scale, "inputs": inputs, "compliance": compliance_rows,
                    "crack_rows": crack_tbl, "crack_text": crack_text, "ignore_dist": ignore_dist,
                    "notes": notes})
            st.session_state.report = {"sig": report_sig, "pdf": pdf}
        rep = st.session_state.get("report")
        if rep and rep["sig"] == report_sig:
            st.sidebar.download_button("Baixar relatório PDF", rep["pdf"], file_name="relatorio_FEA.pdf",
                                       mime="application/pdf", key="dl_report")
