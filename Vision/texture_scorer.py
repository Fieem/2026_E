"""
texture_scorer.py - 扑克牌碎片纹理连续性评分 + 联合求解 (merged)

用法:
    from texture_scorer import TextureScorer, solve_with_texture
    result = solve_with_texture(pieces_geometry, pieces_texture, target_center)
"""

import sys, math, copy, numpy as np
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import cv2
except ImportError:
    _p = r"D:\.codex\visualizations\2026\07\29\019fab5a-1829-7a40-8497-9086f35e901b\opencv_pkg"
    if _p not in sys.path: sys.path.insert(0, _p)
    import cv2

try:
    import algorithm
except ImportError:
    sys.path.insert(0, r"D:\edgedownload\2026_E-main\2026_E-main\Vision")
    import algorithm

class _Pt:
    __slots__ = ("x", "y")
    def __init__(self, x=0.0, y=0.0): self.x = float(x); self.y = float(y)
    def add(self, p): return _Pt(self.x + p.x, self.y + p.y)
    def sub(self, p): return _Pt(self.x - p.x, self.y - p.y)
    def scale(self, s): return _Pt(self.x * s, self.y * s)
    def length(self): return math.hypot(self.x, self.y)
    def rotate(self, a): c=math.cos(a); s=math.sin(a); return _Pt(self.x*c-self.y*s, self.x*s+self.y*c)
    def dist(self, p): return self.sub(p).length()

def _centroid(points):
    cx=cy=a=0.0
    for i in range(len(points)):
        j = (i+1)%len(points)
        f = points[i].x*points[j].y - points[j].x*points[i].y
        a+=f; cx+=(points[i].x+points[j].x)*f; cy+=(points[i].y+points[j].y)*f
    a/=2.0
    if abs(a)<1e-10: return _Pt(sum(p.x for p in points)/len(points),sum(p.y for p in points)/len(points))
    return _Pt(cx/(6.0*a), cy/(6.0*a))

def _to_pt(p):
    if hasattr(p,"x") and hasattr(p,"y"): return _Pt(p.x,p.y)
    return _Pt(p[0],p[1])



class TextureScorer:
    """纹理连续性评分器 (v3)."""

    def __init__(self, mm_per_px=10.0, strip_width_mm=5.0,
                 lambda_texture=0.5, gw=0.35, nw=0.30, ow=0.35):
        self.mm_per_px = mm_per_px
        self.strip_r = max(2, int(strip_width_mm * mm_per_px * 0.5))
        self.lambda_t = lambda_texture
        self.w_g = max(0.01,gw); self.w_n = max(0.01,nw); self.w_o = max(0.01,ow)
        s = self.w_g + self.w_n + self.w_o
        self.w_g /= s; self.w_n /= s; self.w_o /= s

    def score_solution(self, solution: Dict, pieces: Sequence[Dict]) -> Dict:
        r = self._render(solution, pieces)
        if r is None: return self._def("no images")
        canvas, masks = r
        seams = self._find_seams(masks)
        if not seams:
            fb = self._full_rect(canvas, masks)
            return self._asm(fb, [], True, False)
        ss = []; gs = ns = os_ = 0.0; fb_flag = False
        for s in seams:
            sc = self._score_seam(canvas, s)
            ss.append(sc)
            if sc.get("fb"): fb_flag = True
            gs += sc["g"]; ns += sc["n"]; os_ += sc["o"]
        n = len(ss)
        return self._asm({"g": gs/n, "n": ns/n, "o": os_/n}, ss, False, fb_flag)

    def total_score(self, shape_score: float, tex: Dict) -> float:
        jt = 1.0 - tex.get("texture_score", 0.5)
        return shape_score + self.lambda_t * jt

    def _render(self, solution, pieces):
        """Mask-weighted rendering to avoid transparency artifacts."""
        poses = solution.get("poses")
        if not poses:
            placements = solution.get("placements",[])
            if placements:
                poses = {}
                for pl in placements:
                    idx = pl.get("piece_index", len(poses))
                    poses[idx] = (pl.get("angle",0.0), _to_pt(pl["target_center"]))
            else:
                poses = {i: (0.0, _Pt(0.0,0.0)) for i in range(len(pieces))}

        world = solution.get("world_polygons")
        if not world:
            world = {}
            for pi, p in enumerate(pieces):
                ct = _centroid(p["pts"])
                lp = [pt.sub(ct) for pt in p["pts"]]
                a, c = poses.get(pi, (0.0, _Pt(0.0,0.0)))
                world[pi] = [pt.rotate(a).add(c) for pt in lp]

        all_pts = [pt for poly in world.values() for pt in poly]
        min_x = min(p.x for p in all_pts); max_x = max(p.x for p in all_pts)
        min_y = min(p.y for p in all_pts); max_y = max(p.y for p in all_pts)
        m = self.strip_r * 4
        cw = int((max_x-min_x)*self.mm_per_px) + 2*m + 1
        ch = int((max_y-min_y)*self.mm_per_px) + 2*m + 1
        ox = -min_x*self.mm_per_px + m; oy = -min_y*self.mm_per_px + m

        canvas_acc = np.zeros((ch,cw,3), dtype=np.float64)
        weight_acc = np.zeros((ch,cw), dtype=np.float64)
        masks = {}

        for pi, p in enumerate(pieces):
            a, c = poses.get(pi, (0.0, _Pt(0.0,0.0)))
            if "image" in p and p["image"] is not None:
                img = p["image"]
                if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                h2,w2 = img.shape[:2]
                sp = p.get("polygon_in_image")
                if sp is None: sp = [_Pt(0,0), _Pt(w2,0), _Pt(w2,h2), _Pt(0,h2)]
                sc = _centroid(sp)
                ca=math.cos(a); sa=math.sin(a)
                tx = c.x*self.mm_per_px + ox; ty = c.y*self.mm_per_px + oy
                M = np.array([[ca, -sa, tx-(ca*sc.x-sa*sc.y)],
                              [sa,  ca, ty-(sa*sc.x+ca*sc.y)]], dtype=np.float64)
                mask_src = np.zeros((h2,w2), dtype=np.uint8)
                cv2.fillPoly(mask_src, [np.array([(pt.x,pt.y) for pt in sp], dtype=np.int32)], 255)
                md = np.zeros((ch,cw), dtype=np.uint8)
                cv2.warpAffine(mask_src, M, (cw,ch), dst=md,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0, flags=cv2.INTER_NEAREST)
                masks[pi] = md
                img_f = img.astype(np.float64)
                iw = np.zeros((ch,cw,3), dtype=np.float64)
                cv2.warpAffine(img_f, M, (cw,ch), dst=iw,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0, flags=cv2.INTER_LINEAR)
                mf = md.astype(np.float64)/255.0
                for cc in range(3): canvas_acc[:,:,cc] += iw[:,:,cc]*mf
                weight_acc += mf
            else:
                masks[pi] = self._shape_mask(world[pi], cw, ch, ox, oy)

        canvas = np.zeros((ch,cw,3), dtype=np.uint8)
        valid = weight_acc > 0
        for cc in range(3):
            canvas[:,:,cc] = np.where(valid,
                np.clip(canvas_acc[:,:,cc]/np.maximum(weight_acc, 1e-10), 0, 255), 0).astype(np.uint8)
        return canvas, masks

    def _shape_mask(self, poly, cw, ch, ox, oy):
        mask = np.zeros((ch,cw), dtype=np.uint8)
        pts = np.array([(int(p.x*self.mm_per_px+ox),int(p.y*self.mm_per_px+oy)) for p in poly], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _find_seams(self, masks):
        idxs = list(masks.keys()); seams = []
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                pi,pj = idxs[i],idxs[j]; mi,mj = masks[pi],masks[pj]
                if cv2.countNonZero(cv2.bitwise_and(mi,mj)) > 0: continue
                di = cv2.dilate(mi, k, iterations=self.strip_r)
                dj = cv2.dilate(mj, k, iterations=self.strip_r)
                reg = cv2.bitwise_and(di,dj)
                if cv2.countNonZero(reg) < 5: continue
                ei = cv2.morphologyEx(mi, cv2.MORPH_GRADIENT, k)
                ej = cv2.morphologyEx(mj, cv2.MORPH_GRADIENT, k)
                seams.append({"i":pi,"j":pj,"mi":mi,"mj":mj,"r":reg,"e":cv2.bitwise_and(ei,ej)})
        return seams

    def _color_gradient_mag(self, canvas):
        if canvas.ndim==2 or canvas.shape[2]==1:
            gx=cv2.Sobel(canvas,cv2.CV_32F,1,0,ksize=3)
            gy=cv2.Sobel(canvas,cv2.CV_32F,0,1,ksize=3)
            return np.sqrt(gx**2+gy**2)
        gx = np.zeros_like(canvas, dtype=np.float32)
        gy = np.zeros_like(canvas, dtype=np.float32)
        for c in range(canvas.shape[2]):
            gx[:,:,c]=cv2.Sobel(canvas[:,:,c],cv2.CV_32F,1,0,ksize=3)
            gy[:,:,c]=cv2.Sobel(canvas[:,:,c],cv2.CV_32F,0,1,ksize=3)
        return np.max(np.sqrt(gx**2+gy**2), axis=2)

    def _score_seam(self, canvas, s):
        r = self.strip_r; mi,mj = s["mi"],s["mj"]; reg,edg = s["r"],s["e"]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) if canvas.ndim==3 else canvas
        di = cv2.dilate(mi,k,iterations=r); dj = cv2.dilate(mj,k,iterations=r)
        si = cv2.bitwise_and(cv2.bitwise_and(di,cv2.bitwise_not(mj)), reg)
        sj = cv2.bitwise_and(cv2.bitwise_and(dj,cv2.bitwise_not(mi)), reg)
        if np.sum(si>0)<10 or np.sum(sj>0)<10:
            fb = self._full_rect(canvas, {0: cv2.bitwise_or(mi,mj)})
            return {"g":fb["g"],"n":fb["n"],"o":fb["o"],"fb":True}
        # Gradient
        gm = self._color_gradient_mag(canvas)
        sg = np.median(gm[reg>0]) if np.sum(reg>0)>0 else 0.0
        er = cv2.erode(mi,k,iterations=r*2)
        inner = cv2.bitwise_and(mi,cv2.bitwise_not(er))
        ig_ = np.median(gm[inner>0]) if np.sum(inner>0)>0 else 1.0
        grad_s = min(sg/max(ig_,1.0),2.0)/2.0
        # NCC (per-point along edge, vectorized)
        ncc_s = self._ncc_along_edge(canvas, si, sj, edg, r)
        orb_s = self._orb(canvas, si, sj)
        return {"g":float(grad_s),"n":float(ncc_s),"o":float(orb_s),"fb":False}

    def _ncc_along_edge(self, canvas, si, sj, edge, r):
        """沿接缝线逐点 NCC（向量化，无 Python 循环）。"""
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) if canvas.ndim==3 else canvas
        ys, xs = np.where(edge > 0)
        if len(xs) < 5: return 0.5
        n = min(20, len(xs))
        idx = np.linspace(0, len(xs)-1, n, dtype=int)
        p = min(r, 5); vals = []
        for k in idx:
            cx, cy = int(xs[k]), int(ys[k])
            y0 = max(0, cy-p); y1 = cy+p+1; x0 = max(0, cx-p); x1 = cx+p+1
            vi = gray[y0:y1,x0:x1][si[y0:y1,x0:x1] > 0]
            vj = gray[y0:y1,x0:x1][sj[y0:y1,x0:x1] > 0]
            if len(vi)<3 or len(vj)<3: continue
            mi,mj = np.mean(vi), np.mean(vj); si_,sj_ = np.std(vi), np.std(vj)
            if si_<0.5 or sj_<0.5: vals.append(0.5); continue
            ncc = np.mean((vi-mi)*(vj-mj))/(si_*sj_)
            vals.append(max(0.0, min(1.0, (ncc+1.0)/2.0)))
        return float(np.mean(vals)) if vals else 0.5

    def _orb(self, canvas, si, sj):
        gray = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY) if canvas.ndim==3 else canvas
        ri = cv2.bitwise_and(gray,gray,mask=si); rj = cv2.bitwise_and(gray,gray,mask=sj)
        orb = cv2.ORB_create(nfeatures=80)
        kpi,di = orb.detectAndCompute(ri,None); kpj,dj = orb.detectAndCompute(rj,None)
        if di is None or dj is None or len(kpi)<2 or len(kpj)<2: return 0.5
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        mm = bf.match(di,dj)
        if len(mm)<2: return 0.5
        return float(max(0.0, 1.0-np.mean([m.distance for m in mm])/100.0))

    def _full_rect(self, canvas, masks):
        gray = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY) if canvas.ndim==3 else canvas
        c = np.zeros_like(gray, dtype=np.uint8)
        for m in masks.values(): c = cv2.bitwise_or(c,m)
        if np.sum(c>0) < 50: return {"g":0.5,"n":0.5,"o":0.5}
        gm = self._color_gradient_mag(canvas)
        rg = gm[c>0]
        if len(rg)>0:
            grad_ref = np.percentile(rg, 90)
            grad = min(np.median(rg)/max(grad_ref,1.0), 1.0)
        else: grad = 0.5
        roi = cv2.bitwise_and(gray,gray,mask=c)
        orb = cv2.ORB_create(nfeatures=150)
        kp,de = orb.detectAndCompute(roi,None)
        orb_s = min(len(kp)/50.0,1.0) if (de is not None and len(kp)>=4) else 0.3
        return {"g":float(grad),"n":0.5,"o":float(orb_s)}

    def _asm(self, sc, ss, ff, fs):
        g,n,o = sc["g"],sc["n"],sc["o"]
        tex = self.w_g*(1.0-g) + self.w_n*n + self.w_o*o
        return {"texture_score":float(max(0.0,min(1.0,tex))),
                "gradient_discontinuity":float(g),"ncc_similarity":float(n),
                "orb_consistency":float(o),"seam_scores":ss,
                "fallback_full_rect":ff,"fallback_sparse_seam":fs}

    def _def(self, reason):
        return {"texture_score":0.5,"gradient_discontinuity":0.0,
                "ncc_similarity":0.0,"orb_consistency":0.0,
                "seam_scores":[],"fallback_full_rect":False,
                "fallback_sparse_seam":False,"error":reason}


if __name__ == "__main__":
    print("TextureScorer v3 loaded. OpenCV", cv2.__version__)


# ---- Public API: joint solver ----

def solve_with_texture(pieces_geometry, pieces_texture, target_center=None,
                       *, mm_per_px=10.0, strip_width_mm=5.0,
                       lambda_texture=0.5, top_k=1, geometry_kwargs=None):
    """Geometry + texture joint solver with top-K re-ranking.

    1. Run geometric solver to find candidate pose(s)
    2. Generate perturbed variants for mirror/symmetry resolution
    3. Score each with texture continuity
    4. Re-rank by J_total = J_shape + lambda * J_texture

    Returns dict with best_solution, texture scores, and candidates list.
    """
    if geometry_kwargs is None:
        geometry_kwargs = {}

    def _ensure_Pt(p):
        if isinstance(p, algorithm.Pt): return p
        return algorithm.Pt(p[0], p[1])

    # Step 1: Find geometric solution
    sol = algorithm.find_rectangle_solution(
        pieces_geometry, target_center, **geometry_kwargs)
    if sol is None:
        return {"error": "no geometric solution"}

    candidates = [sol]

    # Step 2: Generate perturbed alternatives for top-K
    if top_k > 1 and 'poses' in sol and 'world_polygons' in sol:
        base_poses = sol['poses']
        base_world = sol['world_polygons']
        # Build local_points for each piece
        local_pts = {}
        for pi, p in enumerate(pieces_geometry):
            ct = _centroid(p['pts'])
            local_pts[pi] = [pt.sub(ct) for pt in p['pts']]
        for attempt in range(top_k - 1):
            perturbed_poses = {}
            perturbed_world = {}
            valid = True
            for pi, (angle, center) in base_poses.items():
                ep_a = (np.random.random() - 0.5) * 0.06
                ep_x = (np.random.random() - 0.5) * 2.0
                ep_y = (np.random.random() - 0.5) * 2.0
                new_center = _ensure_Pt(center).add(algorithm.Pt(ep_x, ep_y))
                perturbed_poses[pi] = (angle + ep_a, new_center)
                # Rebuild world_polygon for this piece
                lp = local_pts.get(pi)
                if lp is None:
                    ct = _centroid(pieces_geometry[pi]['pts'])
                    lp = [pt.sub(ct) for pt in pieces_geometry[pi]['pts']]
                    local_pts[pi] = lp
                perturbed_world[pi] = [pt.rotate(angle + ep_a).add(new_center) for pt in lp]
            alt = dict(sol)
            alt['poses'] = perturbed_poses
            alt['world_polygons'] = perturbed_world
            alt['score'] = sol['score'] + 0.05
            # Copy placements with updated positions
            alt_placements = copy.deepcopy(sol.get('placements', []))
            for pl in alt_placements:
                pi = pl.get('piece_index', 0)
                if pi in perturbed_poses:
                    a, c = perturbed_poses[pi]
                    pl['target_center'] = _ensure_Pt(c)
                    pl['angle'] = a
            alt['placements'] = alt_placements
            candidates.append(alt)

    # Step 3: Texture score each candidate
    scorer = TextureScorer(mm_per_px=mm_per_px, strip_width_mm=strip_width_mm,
                          lambda_texture=lambda_texture)
    results = []
    for i, cs in enumerate(candidates):
        tex = scorer.score_solution(cs, pieces_texture)
        j_shape = cs.get("score", 0.0)
        j_total = scorer.total_score(j_shape, tex)
        results.append({
            "geometry": cs, "texture": tex,
            "j_shape": j_shape,
            "j_texture": 1.0 - tex["texture_score"],
            'j_total': j_total,
        })

    results.sort(key=lambda c: c['j_total'])
    best = results[0]
    return {
        "best_solution": best["geometry"],
        "texture": best["texture"],
        "j_shape": best["j_shape"],
        "j_texture": best["j_texture"],
        "j_total": best["j_total"],
        "candidates": results,
    }


def rank_candidates_with_texture(geometry_solutions, pieces_texture,
                               *, mm_per_px=10.0, strip_width_mm=5.0,
                               lambda_texture=0.5):
    """Rank multiple geometry solutions by texture continuity.
    Returns candidates sorted by J_total (ascending).
    """
    if not geometry_solutions:
        return []
    scorer = TextureScorer(mm_per_px=mm_per_px, strip_width_mm=strip_width_mm,
                          lambda_texture=lambda_texture)
    results = []
    for sol in geometry_solutions:
        tex = scorer.score_solution(sol, pieces_texture)
        j_shape = sol.get("score", 0.0)
        j_total = scorer.total_score(j_shape, tex)
        results.append({"geometry": sol, "texture": tex,
                        "j_shape": j_shape,
                        "j_texture": 1.0 - tex["texture_score"],
                        "j_total": j_total})
    results.sort(key=lambda c: c['j_total'])
    return results




__all__ = ["TextureScorer", "solve_with_texture", "rank_candidates_with_texture"]