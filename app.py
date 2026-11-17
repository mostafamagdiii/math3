import base64
import json
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
import mpmath as mp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
X = sp.Symbol("x")
SAFE = {"x": X, "e": sp.E, "pi": sp.pi, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
        "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational}


# ---------------------------------------------------------------- linear systems
def jacobi(A, b, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    D = np.diag(A)
    R = A - np.diagflat(D)
    for k in range(1, max_iter + 1):
        x_new = (b - R @ x) / D
        if not np.all(np.isfinite(x_new)) or np.max(np.abs(x_new)) > 1e100:
            return x_new, k, False
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k, True
        x = x_new
    return x, max_iter, False


def gauss_seidel(A, b, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s = A[i, :i] @ x[:i] + A[i, i + 1:] @ x_old[i + 1:]
            x[i] = (b[i] - s) / A[i, i]
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e100:
            return x, k, False
        if np.max(np.abs(x - x_old)) < tol:
            return x, k, True
    return x, max_iter, False


# ---------------------------------------------------------------- root finding
def bisection(f, a, b, tol, max_iter):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None, 0, False
    for k in range(1, max_iter + 1):
        m = (a + b) / 2
        fm = f(m)
        if abs(fm) == 0 or (b - a) / 2 < tol:
            return m, k, True
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return (a + b) / 2, max_iter, False


def newton(f, df, x0, tol, max_iter):
    x = x0
    for k in range(1, max_iter + 1):
        d = df(x)
        if d == 0 or not np.isfinite(d):
            return x, k, False
        x_new = x - f(x) / d
        if not np.isfinite(x_new):
            return x_new, k, False
        if abs(x_new - x) < tol:
            return x_new, k, True
        x = x_new
    return x, max_iter, False


def secant(f, x0, x1, tol, max_iter):
    for k in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)
        if f1 == f0:
            return x1, k, False
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if not np.isfinite(x2):
            return x2, k, False
        if abs(x2 - x1) < tol:
            return x2, k, True
        x0, x1 = x1, x2
    return x1, max_iter, False


def reference_root(f_sym, a, b):
    """High-precision reference root (bisection in mpmath, 50 digits)."""
    mp.mp.dps = 50
    g = sp.lambdify(X, f_sym, "mpmath")
    lo, hi = mp.mpf(a), mp.mpf(b)
    glo = g(lo)
    for _ in range(200):
        mid = (lo + hi) / 2
        gm = g(mid)
        if glo * gm <= 0:
            hi = mid
        else:
            lo, glo = mid, gm
    return float((lo + hi) / 2)


def timed(fn, *args):
    t = time.perf_counter()
    out = fn(*args)
    return out, (time.perf_counter() - t) * 1000


def pick_best(df):
    ok = df[df["converged"]]
    if ok.empty:
        return None
    return ok.sort_values(["error", "iterations", "time_ms"]).iloc[0]["method"]


# ---------------------------------------------------------------- file / image input
MODE_SYS = "نظام معادلات خطية (Jacobi / Gauss-Seidel)"
MODE_ROOT = "معادلة واحدة / إيجاد جذر (Bisection / Newton / Secant)"
DEFAULTS = {"mode": MODE_SYS, "a_txt": "10 -1 2\n-1 11 -1\n2 -1 10", "b_txt": "6 25 -11",
            "f_txt": "x^3 - x - 2", "a_val": 1.0, "b_val": 2.0}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

EXTRACT_PROMPT = """You are reading a numerical-methods exercise from an image or document.
Return ONLY a JSON object, no prose, no code fences.

If it is a system of linear equations Ax = b:
{"type": "system", "A": [[...], ...], "b": [...]}
(rewrite the equations so every row has coefficients for x1..xn in order, put 0 for missing terms)

If it is a single equation / root-finding problem f(x) = 0:
{"type": "root", "f": "<expression in x, python/sympy syntax, use ** or ^ for powers>", "a": <number or null>, "b": <number or null>}
(move everything to one side so the equation is f(x) = 0; a and b are the interval if given, else null)

If you cannot find such a problem: {"type": "none"}"""


def matrix_to_texts(M):
    M = np.atleast_2d(np.array(M, dtype=float))
    n = M.shape[0]
    if M.shape[1] != n + 1:
        raise ValueError(f"المصفوفة لازم تكون [A | b] يعني {n} صفوف و {n + 1} أعمدة، الموجود {M.shape}")
    a_txt = "\n".join(" ".join(format(v, ".12g") for v in row[:-1]) for row in M)
    b_txt = " ".join(format(v, ".12g") for v in M[:, -1])
    return a_txt, b_txt


def read_uploaded_table(file):
    """CSV / TXT / XLSX -> ('matrix', ndarray) or ('text', str)."""
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return "matrix", pd.read_excel(file, header=None).to_numpy(dtype=float)
    text = file.getvalue().decode("utf-8", errors="ignore").strip()
    try:
        rows = [[float(v) for v in ln.replace(",", " ").replace(";", " ").split()]
                for ln in text.splitlines() if ln.strip()]
        return "matrix", np.array(rows, dtype=float)
    except ValueError:
        return "text", text


def get_secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


def extract_with_gemini(file):
    import requests
    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("مفيش GEMINI_API_KEY في الـ Secrets")
    primary = get_secret("GEMINI_MODEL", "gemini-3.5-flash")
    models = [primary] + [m for m in ("gemini-2.5-flash", "gemini-flash-latest") if m != primary]
    mime = "application/pdf" if file.name.lower().endswith(".pdf") else file.type
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": mime, "data": base64.standard_b64encode(file.getvalue()).decode()}},
            {"text": EXTRACT_PROMPT}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    last_err = "unknown error"
    for model in models:
        for attempt in range(3):
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json=body, timeout=90)
            if r.status_code == 200:
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text[text.index("{"): text.rindex("}") + 1])
            last_err = f"{model}: {r.status_code} {r.text[:200]}"
            if r.status_code in (429, 500, 503):   # busy: wait and retry, then try next model
                time.sleep(2 * (attempt + 1))
                continue
            break                                  # 400/403/404...: try next model
    raise RuntimeError(f"Gemini مش متاح دلوقتي، جرّب بعد شوية. آخر خطأ: {last_err}")


def apply_extracted(d):
    if d.get("type") == "system":
        M = np.hstack([np.array(d["A"], dtype=float), np.array(d["b"], dtype=float).reshape(-1, 1)])
        st.session_state["a_txt"], st.session_state["b_txt"] = matrix_to_texts(M)
        st.session_state["mode"] = MODE_SYS
        return "اتقرا نظام معادلات خطية"
    if d.get("type") == "root":
        st.session_state["f_txt"] = str(d["f"]).replace("**", "^")
        st.session_state["mode"] = MODE_ROOT
        if d.get("a") is not None and d.get("b") is not None:
            st.session_state["a_val"], st.session_state["b_val"] = float(d["a"]), float(d["b"])
            return "اتقرت معادلة والفترة"
        return "اتقرت معادلة، بس الفترة مش مكتوبة، حددها بنفسك تحت"
    raise ValueError("ملقيتش مسألة واضحة في الملف")


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Numerical Methods Comparator", page_icon="🧮", layout="wide")
st.title("🧮 Numerical Methods Comparator")
st.caption("جرّب أكتر من طريقة على نفس المسألة وشوف أدقّها وأسرعها")

with st.expander("📎 ارفع صورة السؤال أو ملف (اختياري)"):
    st.caption("صورة / PDF → بيتقرا بالذكاء الاصطناعي. CSV / TXT / Excel فيه المصفوفة [A | b] → بيتقرا مباشرة. "
               "راجع المدخلات اللي هتظهر تحت قبل ما تشغّل.")
    up = st.file_uploader("الملف", type=["png", "jpg", "jpeg", "webp", "pdf", "txt", "csv", "xlsx"])
    if up is not None:
        if up.type.startswith("image/"):
            st.image(up, width=360)
        if st.button("استخرج السؤال"):
            try:
                if up.name.lower().endswith((".txt", ".csv", ".xlsx", ".xls")):
                    kind, val = read_uploaded_table(up)
                    if kind == "matrix":
                        st.session_state["a_txt"], st.session_state["b_txt"] = matrix_to_texts(val)
                        st.session_state["mode"] = MODE_SYS
                        st.success("اتقرا نظام معادلات خطية من الملف")
                    else:
                        st.session_state["f_txt"] = val.splitlines()[0].replace("**", "^")
                        st.session_state["mode"] = MODE_ROOT
                        st.success("اتقرت الدالة من الملف")
                else:
                    with st.spinner("بقرا السؤال..."):
                        st.success(apply_extracted(extract_with_gemini(up)))
            except Exception as e:
                st.error(f"معرفتش أقرا الملف: {e}")

mode = st.radio("نوع المسألة", [MODE_SYS, MODE_ROOT], key="mode")
c1, c2 = st.columns(2)
tol = c1.number_input("Tolerance", value=1e-8, format="%.0e")
max_iter = c2.number_input("Max iterations", value=500, min_value=1, step=50)

if mode == MODE_SYS:
    st.subheader("المصفوفة A والمتجه b")
    a_txt = st.text_area("المصفوفة A (كل صف في سطر، القيم بمسافة أو فاصلة)", key="a_txt", height=120)
    b_txt = st.text_input("المتجه b (بمسافة أو فاصلة)", key="b_txt")
    if st.button("شغّل الطرق", type="primary"):
        try:
            A = np.array([[float(v) for v in r.replace(",", " ").split()]
                          for r in a_txt.strip().splitlines() if r.strip()])
            b = np.array([float(v) for v in b_txt.replace(",", " ").split()])
            n = len(b)
            assert A.shape == (n, n), "A لازم تكون مربعة وبنفس حجم b"
            assert np.all(np.diag(A) != 0), "فيه صفر على القطر الرئيسي، رتّب المعادلات"
            exact = np.linalg.solve(A, b)

            dd = bool(np.all(np.abs(np.diag(A)) >= np.sum(np.abs(A), axis=1) - np.abs(np.diag(A))))
            D, L, U = np.diag(np.diag(A)), np.tril(A, -1), np.triu(A, 1)
            rho_j = max(abs(np.linalg.eigvals(-np.linalg.inv(D) @ (L + U))))
            rho_gs = max(abs(np.linalg.eigvals(-np.linalg.inv(D + L) @ U)))

            rows = []
            for name, fn in [("Jacobi", jacobi), ("Gauss-Seidel", gauss_seidel)]:
                (x, it, ok), ms = timed(fn, A, b, tol, int(max_iter))
                rows.append({"method": name, "converged": ok, "iterations": it,
                             "error": float(np.max(np.abs(x - exact))),
                             "residual": float(np.max(np.abs(A @ x - b))),
                             "time_ms": round(ms, 3), "solution": np.round(x, 6).tolist()})
            df = pd.DataFrame(rows)
            st.success(f"الحل الدقيق (numpy): {np.round(exact, 6).tolist()}")
            st.dataframe(df, use_container_width=True)
            best = pick_best(df)
            st.markdown(f"### 🏆 الأفضل: **{best}**" if best else "### ⚠️ ولا طريقة تقاربت")
            st.info(f"Diagonally dominant: {'نعم' if dd else 'لا'} | "
                    f"Spectral radius → Jacobi: {rho_j:.4f}, Gauss-Seidel: {rho_gs:.4f} "
                    "(أقل من 1 معناها إن الطريقة تتقارب)")
        except Exception as e:
            st.error(f"خطأ في المدخلات: {e}")
else:
    st.subheader("الدالة f(x) = 0")
    f_txt = st.text_input("f(x)  (مثال: x^3 - x - 2  أو  cos(x) - x)", key="f_txt")
    c3, c4 = st.columns(2)
    a = c3.number_input("بداية الفترة a", key="a_val")
    b = c4.number_input("نهاية الفترة b", key="b_val")
    if st.button("شغّل الطرق", type="primary"):
        try:
            f_sym = parse_expr(f_txt, local_dict=SAFE, global_dict={}, transformations=TRANSFORMS)
            f = sp.lambdify(X, f_sym, "math")
            df_ = sp.lambdify(X, sp.diff(f_sym, X), "math")
            ref = reference_root(f_sym, a, b)
            if f(a) * f(b) > 0:
                st.warning("f(a) و f(b) لهم نفس الإشارة، فالـ Bisection مش هتشتغل. غيّر الفترة.")
            rows = []
            runs = [("Bisection", bisection, (f, a, b)),
                    ("Newton", newton, (f, df_, (a + b) / 2)),
                    ("Secant", secant, (f, a, b))]
            for name, fn, args in runs:
                try:
                    (x, it, ok), ms = timed(fn, *args, tol, int(max_iter))
                except Exception:
                    x, it, ok, ms = None, 0, False, 0.0
                if x is None:
                    rows.append({"method": name, "converged": False, "iterations": it,
                                 "error": float("inf"), "residual": float("inf"),
                                 "time_ms": 0.0, "root": None})
                else:
                    rows.append({"method": name, "converged": ok, "iterations": it,
                                 "error": abs(x - ref), "residual": abs(f(x)),
                                 "time_ms": round(ms, 3), "root": x})
            df = pd.DataFrame(rows)
            st.success(f"الجذر المرجعي (دقة عالية): {ref:.12f}")
            st.dataframe(df, use_container_width=True)
            best = pick_best(df)
            st.markdown(f"### 🏆 الأفضل: **{best}**" if best else "### ⚠️ ولا طريقة تقاربت")
            st.caption("Newton بيبدأ من منتصف الفترة، و Secant بيبدأ من a و b.")
        except Exception as e:
            st.error(f"خطأ في الدالة أو المدخلات: {e}")
