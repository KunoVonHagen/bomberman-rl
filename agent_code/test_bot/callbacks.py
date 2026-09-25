import time

import numpy as np

try:
    import settings as _s
    BOMB_TIMER = int(_s.BOMB_TIMER)
    BOMB_POWER = int(_s.BOMB_POWER)
    EXPLOSION_TIMER = int(_s.EXPLOSION_TIMER)
    TOTAL_COINS = int(_s.SCENARIOS["classic"]["COIN_COUNT"])
except Exception:
    BOMB_TIMER, BOMB_POWER, EXPLOSION_TIMER, TOTAL_COINS = 4, 3, 2, 9

FUSE = BOMB_TIMER
EXP = EXPLOSION_TIMER
COOLDOWN = FUSE + EXP + 1
MOVES4 = (("UP", (0, -1)), ("DOWN", (0, 1)), ("LEFT", (-1, 0)), ("RIGHT", (1, 0)))
DELTA = dict(MOVES4)
DELTA["WAIT"] = (0, 0)

HORIZON = 40
TIME_BUDGET = 0.22
BOMB_OVERHEAD = 5.0
KILL_VALUE = 5.0
HUNT_WEIGHT = 0.012
SOFT_PENALTY = 0.4
OPP_RACE_LOSE = 0.10
OPP_RACE_TIE = 0.55
LENIENT_KILL_FACTOR = 0.6
NAIVE_ALPHA = 0.20
NAIVE_DECAY = 0.85


class _Memory:
    def __init__(self):
        self.round = -1
        self.seen_coins = set()
        self.last_drop_step = None
        self.bcache = {}
        self.rng = np.random.default_rng()


def setup(self):
    self.mem = _Memory()


def act(self, game_state: dict):
    mem = getattr(self, "mem", None)
    if mem is None:
        mem = self.mem = _Memory()
    if game_state["round"] != mem.round:
        mem.round = game_state["round"]
        mem.seen_coins = set()
        mem.last_drop_step = None
        mem.bcache = {}
    try:
        action = Planner(game_state, mem, time.time() + TIME_BUDGET).decide()
    except Exception as ex:
        try:
            self.logger.exception(ex)
        except Exception:
            pass
        action = "WAIT"
    if action == "BOMB":
        mem.last_drop_step = game_state["step"]
    return action


def setup_training(self):
    pass


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    pass


def end_of_round(self, last_game_state, last_action, events):
    pass


class Planner:
    def __init__(self, gs, mem, deadline):
        self.gs, self.mem, self.deadline = gs, mem, deadline
        self.field = gs["field"]
        self.step = gs["step"]
        _, _, bombs_left, pos = gs["self"]
        self.me = (int(pos[0]), int(pos[1]))
        self.bombs_left = bool(bombs_left)
        self.others = [(o[0], bool(o[2]), (int(o[3][0]), int(o[3][1]))) for o in gs["others"]]
        self.opp_cells = {o[2] for o in self.others}
        self.coins = [(int(c[0]), int(c[1])) for c in gs["coins"]]
        self.bombs = [((int(b[0][0]), int(b[0][1])), int(b[1])) for b in gs["bombs"]]
        self.bomb_cells = {b[0] for b in self.bombs}

        xs, ys = np.nonzero(self.field == 0)
        self.free_cells = list(zip(xs.tolist(), ys.tolist()))
        self.free_set = set(self.free_cells)
        self.crates = int((self.field == 1).sum())
        mem.seen_coins |= set(self.coins)
        hidden = max(0, TOTAL_COINS - len(mem.seen_coins))
        self.crate_value = min(0.5, max(0.02, hidden / max(self.crates, 1)))

        self._of = None
        self._opp_dist = None
        self._reach = None
        self.nbo = None
        self._soft_cache = {}

    def blast(self, c):
        b = self.mem.bcache.get(c)
        if b is None:
            x, y = c
            cells = [c]
            f = self.field
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                for i in range(1, BOMB_POWER + 1):
                    nx, ny = x + i * dx, y + i * dy
                    if f[nx, ny] == -1:
                        break
                    cells.append((nx, ny))
            b = (cells, frozenset(cells))
            self.mem.bcache[c] = b
        return b

    @staticmethod
    def _t_end(danger):
        return max(2, max((m.bit_length() for m in danger.values()), default=0))

    def build_danger(self):
        danger = {}
        for pos, timer in self.bombs:
            m = 0
            for d in range(EXP):
                m |= 1 << (timer + 1 + d)
            for c in self.blast(pos)[0]:
                danger[c] = danger.get(c, 0) | m
        em = self.gs["explosion_map"]
        xs, ys = np.nonzero(em > 0)
        for x, y in zip(xs.tolist(), ys.tolist()):
            m = 0
            for t in range(1, int(em[x, y]) + 1):
                m |= 1 << t
            danger[(x, y)] = danger.get((x, y), 0) | m
        self.danger = danger
        self.T_end = self._t_end(danger)

    def build_hyp_danger(self):
        """Danger if every armed opponent near me drops a bomb right now."""
        hyp = None
        for _, armed, pos in self.others:
            if not armed or abs(pos[0] - self.me[0]) + abs(pos[1] - self.me[1]) > 7:
                continue
            if pos in self.bomb_cells:
                continue
            if hyp is None:
                hyp = dict(self.danger)
            m = 0
            for d in range(EXP):
                m |= 1 << (FUSE + 1 + d)
            for c in self.blast(pos)[0]:
                hyp[c] = hyp.get(c, 0) | m
        self.danger_h = hyp
        self.T_end_h = self._t_end(hyp) if hyp else self.T_end

    def make_graph(self, blocked):
        tgt = self.free_set - blocked
        cells = list(tgt)
        if self.me not in tgt:
            cells.append(self.me)
        nb = {}
        for c in cells:
            x, y = c
            nb[c] = [(a, (x + dx, y + dy)) for a, (dx, dy) in MOVES4 if (x + dx, y + dy) in tgt]
        return cells, nb

    @staticmethod
    def viability(cells, nb, danger, T_end, R=None):
        V = [None] * (T_end + 1)
        V[T_end] = {c for c in cells if not (danger.get(c, 0) >> T_end) & 1}
        for t in range(T_end - 1, -1, -1):
            nxt = V[t + 1]
            cur = set()
            blk = R[t + 1] if (R is not None and t + 1 < len(R)) else ()
            for c in cells:
                if (danger.get(c, 0) >> t) & 1:
                    continue
                if c in nxt:
                    cur.add(c)
                    continue
                for _, n in nb[c]:
                    if n in nxt and n not in blk:
                        cur.add(c)
                        break
            V[t] = cur
        return V

    def forward(self, V, cellset, R=None):
        me, T_end = self.me, self.T_end
        layers = [{me: None}]
        arrive = {me: (0, None)}
        for t in range(1, HORIZON + 1):
            S = V[t] if t <= T_end else cellset
            prev = layers[-1]
            cur = {}
            blk = R[t] if (R is not None and t < len(R)) else ()
            for c, fa in prev.items():
                if c in S and c not in cur:
                    cur[c] = fa if fa else "WAIT"
                for a, n in self.nb[c]:
                    if n in S and n not in cur and n not in blk:
                        cur[n] = fa if fa else a
            layers.append(cur)
            for c, fa in cur.items():
                if c not in arrive:
                    arrive[c] = (t, fa)
            if t > T_end and len(cur) == len(prev):
                break
        self.layers, self.arrive = layers, arrive

    def layer(self, t):
        return self.layers[min(t, len(self.layers) - 1)]

    def opp_graph(self):
        if self.nbo is None:
            tgt = self.free_set - self.bomb_cells
            nbo = {}
            for c in self.free_cells:
                x, y = c
                lst = [c]
                for _, (dx, dy) in MOVES4:
                    n = (x + dx, y + dy)
                    if n in tgt:
                        lst.append(n)
                nbo[c] = lst
            self.nbo = nbo
        return self.nbo

    def opp_dist(self):
        if self._opp_dist is None:
            nbo = self.opp_graph()
            dist, frontier = {}, []
            for p in self.opp_cells:
                dist[p] = 0
                frontier.append(p)
            d = 0
            while frontier:
                d += 1
                nxt = []
                for c in frontier:
                    for n in nbo.get(c, ()):
                        if n not in dist:
                            dist[n] = d
                            nxt.append(n)
                frontier = nxt
            self._opp_dist = dist
        return self._opp_dist

    def opp_floods(self):
        if self._of is None:
            nbo = self.opp_graph()
            H = 4 + FUSE + EXP
            res = []
            for _, _, pos in self.others:
                layers = [{pos}]
                doomed = False
                for t in range(1, H + 1):
                    cur = set()
                    for c in layers[-1]:
                        for n in nbo.get(c, (c,)):
                            if not (self.danger.get(n, 0) >> t) & 1:
                                cur.add(n)
                    if not cur:
                        doomed = True
                        break
                    layers.append(cur)
                res.append(None if doomed else layers)
            self._of = res
        return self._of

    def opp_reach(self):
        """R[t] = cells some opponent could stand on at time t."""
        if self._reach is None:
            floods = [f for f in self.opp_floods() if f is not None]
            R = []
            for t in range(4 + FUSE + EXP + 1):
                u = set()
                for f in floods:
                    if t < len(f):
                        u |= f[t]
                R.append(u)
            self._reach = R
        return self._reach

    def kill_prob(self, X, tau):
        _, bset = self.blast(X)
        m_times = range(tau + FUSE + 1, tau + FUSE + EXP + 1)
        last_t = tau + FUSE + EXP
        nbo = self.opp_graph()
        of = self.opp_floods()
        total = 0.0
        for oi, (_, _, pos) in enumerate(self.others):
            base = of[oi]
            if base is None or tau >= len(base):
                continue
            if abs(pos[0] - X[0]) + abs(pos[1] - X[1]) > 8:
                continue
            if not any(t < len(base) and (base[t] & bset) for t in m_times):
                continue
            cur = set(base[tau])
            cur.discard(X)
            sizes = []
            killed_by_me = False
            for t in range(tau + 1, last_t + 1):
                nxt = set()
                mine = t in m_times
                for c in cur:
                    for n in nbo.get(c, (c,)):
                        if n == X:
                            continue
                        if (self.danger.get(n, 0) >> t) & 1:
                            continue
                        if mine and n in bset:
                            continue
                        nxt.add(n)
                cur = nxt
                if mine:
                    sizes.append(len(cur))
                if not cur:
                    killed_by_me = t in m_times
                    break
            if killed_by_me:
                total += 1.0
            elif cur and sizes:
                w = min(sizes)
                if w <= 4:
                    total += 0.7 / (1 + w)
        return total

    def escape_ok(self, X, tau, danger, T_end, strict=False):
        """After dropping a bomb at X from state tau, is there still a way out?
        strict: opponents may step on any cell they can reach in time."""
        R = self.opp_reach() if (strict and self.others) else None
        _, bset = self.blast(X)
        m = 0
        for d in range(EXP):
            m |= 1 << (tau + FUSE + 1 + d)
        last_t = max(T_end, tau + FUSE + EXP)
        if (danger.get(X, 0) >> (tau + 1)) & 1:
            return False
        cur = {X}
        nb = self.nb
        for t in range(tau + 2, last_t + 1):
            nxt = set()
            for c in cur:
                for n in [c] + [n for _, n in nb[c] if n != X]:
                    if R is not None and n != c and t < len(R) and n in R[t]:
                        continue
                    d = danger.get(n, 0)
                    if n in bset:
                        d |= m
                    if not (d >> t) & 1:
                        nxt.add(n)
            cur = nxt
            if not cur:
                return False
        return True

    def soft_ok(self, action):
        if self.danger_h is None:
            return True
        r = self._soft_cache.get(action)
        if r is None:
            if action == "BOMB":
                r = self.escape_ok(self.me, 0, self.danger_h, self.T_end_h)
            else:
                dx, dy = DELTA[action]
                r = (self.me[0] + dx, self.me[1] + dy) in self.Vh[1]
            self._soft_cache[action] = r
        return r

    def avail_tau(self):
        if self.bombs_left:
            return 0
        last = self.mem.last_drop_step
        if last is None:
            return 4
        return max(1, last + COOLDOWN - self.step)

    def prepare(self, attempt):
        """1: strict (only when threatened), 2: opponents block their cell, 3: ignore opponents."""
        blocked = set(self.bomb_cells)
        R = None
        if attempt <= 2:
            blocked |= self.opp_cells
        if attempt == 1:
            if not self.others or not self.danger.get(self.me):
                return False
            R = self.opp_reach()
        cells, nb = self.make_graph(blocked)
        self.nb = nb
        cellset = set(cells)
        V = self.viability(cells, nb, self.danger, self.T_end, R)
        if self.me not in V[0]:
            return False
        self.V = V
        self.forward(V, cellset, R)
        self.Vh = None
        if self.danger_h is not None:
            self.Vh = self.viability(cells, nb, self.danger_h, self.T_end_h)
        return True

    def decide(self):
        self.build_danger()
        self.build_hyp_danger()
        ok = False
        for attempt in (1, 2, 3):
            if self.prepare(attempt):
                ok = True
                break
        if not ok:
            return self.panic()

        cands = []
        opp_d = self.opp_dist() if self.opp_cells else {}
        for c in self.coins:
            a = self.arrive.get(c)
            if a is None or a[0] == 0:
                continue
            t, fa = a
            val = 1.0
            od = opp_d.get(c)
            if od is not None:
                if od < t:
                    val = OPP_RACE_LOSE
                elif od == t:
                    val = OPP_RACE_TIE
            cands.append((val / (t + 1.0), fa))

        bomb = self.best_bomb()
        if bomb is not None:
            cands.append(bomb)
        if self.opp_cells:
            cands.extend(self.hunt_candidates(opp_d))

        rng = self.mem.rng
        best, best_u = None, -1.0
        for u, fa in cands:
            if fa is None:
                continue
            if fa != "BOMB" and not self.first_action_ok(fa):
                continue
            if not self.soft_ok(fa):
                u *= SOFT_PENALTY
            u *= 1.0 + 1e-6 * rng.random()
            if u > best_u:
                best, best_u = fa, u
        if best is not None:
            return best
        return self.idle_action()

    def first_action_ok(self, a):
        if a == "WAIT":
            return True
        dx, dy = DELTA[a]
        dest = (self.me[0] + dx, self.me[1] + dy)
        return dest not in self.opp_cells and dest not in self.bomb_cells

    def idle_action(self):
        l1 = self.layers[1] if len(self.layers) > 1 else {}
        if self.me in l1:
            return "WAIT"
        for _, fa in l1.items():
            if self.first_action_ok(fa) and self.soft_ok(fa):
                return fa
        for _, fa in l1.items():
            if self.first_action_ok(fa):
                return fa
        return "WAIT"

    def best_bomb(self):
        avail = self.avail_tau()
        opp_near = bool(self.others)
        f = self.field
        rows = []
        for X, (t0, _) in self.arrive.items():
            if time.time() > self.deadline:
                break
            tau = max(t0, avail)
            tt = None
            for k in range(tau, tau + 3):
                if X in self.layer(k):
                    tt = k
                    break
            if tt is None:
                continue
            cells, bset = self.blast(X)
            crates = sum(1 for (cx, cy) in cells if f[cx, cy] == 1)
            value = crates * self.crate_value
            kp = 0.0
            if opp_near:
                naive = 0.0
                for _, _, pos in self.others:
                    if pos in bset:
                        naive += NAIVE_DECAY ** tt
                value += KILL_VALUE * NAIVE_ALPHA * naive
                if tt <= 4:
                    kp = self.kill_prob(X, tt)
                    value += KILL_VALUE * kp
            if value <= 1e-9:
                continue
            rows.append((value / (tt + BOMB_OVERHEAD), X, tt, kp))
        rows.sort(reverse=True)
        for u, X, tt, kp in rows[:12]:
            if time.time() > self.deadline:
                break
            if not self.escape_ok(X, tt, self.danger, self.T_end, strict=True):
                if kp >= 0.99 and self.escape_ok(X, tt, self.danger, self.T_end):
                    u *= LENIENT_KILL_FACTOR
                else:
                    continue
            fa = "BOMB" if tt == 0 else self.layer(tt)[X]
            return (u, fa)
        return None

    def hunt_candidates(self, opp_d):
        out = []
        for c, (t, fa) in self.arrive.items():
            d = opp_d.get(c)
            if d is None:
                continue
            u = HUNT_WEIGHT / (1.0 + max(d - 2, 0) + 0.2 * t + (2.0 if d < 2 else 0.0))
            if t == 0:
                if self.me not in self.layers[1]:
                    continue
                fa = "WAIT"
            out.append((u, fa))
        return out

    def panic(self):
        best, best_s = "WAIT", -1e9
        for a, (dx, dy) in list(MOVES4) + [("WAIT", (0, 0))]:
            dest = (self.me[0] + dx, self.me[1] + dy)
            if a != "WAIT" and (dest not in self.free_set or dest in self.bomb_cells
                                or dest in self.opp_cells):
                continue
            m = self.danger.get(dest, 0)
            first = 99
            for t in range(1, 12):
                if (m >> t) & 1:
                    first = t
                    break
            s = (0 if (m >> 1) & 1 else 1000) + first * 10 - bin(m).count("1")
            if s > best_s:
                best, best_s = a, s
        return best