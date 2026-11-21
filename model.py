import numpy as np
import pandas as pd

file_path = "FRA_maize_data.xlsx"
df = pd.read_excel(file_path)

WORKBOOK = "FRA_maize_data.xlsx"
RESULTS = "FRA_results.xlsx"
HORIZON = 8
PATHS = 6000
SEED = 11
GRID_R = np.arange(0, 2.51e6, 5e4)
TARGETS = (0.10, 0.05, 0.02, 0.01)
FLOOR_PRICE_PER_TONNE = 347 * 20


def load_parameters(path=WORKBOOK):
    table = pd.read_excel(path, sheet_name="Parameters", nrows=11)
    params = {str(r.iloc[0]).split(" ")[0]: tuple(float(r.iloc[k]) for k in (1, 2, 3)) for _, r in table.iterrows()}
    supply = pd.read_excel(path, sheet_name="Other supply")
    other = supply[pd.to_numeric(supply.iloc[:, 0], errors="coerce").notna()].iloc[:, 10].astype(float)
    positive = other[other > 0]
    params["other_supply"] = (positive.min(), positive.median(), positive.max())
    return params


def load_series(path=WORKBOOK):
    d = pd.read_excel(path, sheet_name="Maize production").iloc[:, :9]
    d.columns = ["year", "season", "cfs", "bought", "target", "req", "carry", "floor", "cap"]
    d = d.sort_values("year").reset_index(drop=True)
    d["log_ratio"] = np.log(d.cfs / d.req)
    return d


def build_case(params, case="base", large_import_cap=False, hold_scale=1.0):
    low_cost, high_cost = {"cheap": (0, 2), "base": (1, 1), "costly": (2, 0)}[case]
    return dict(
        hold=params["hold"][low_cost] * hold_scale,
        rent_extra=params["rent_extra"][low_cost],
        loss_own=params["loss_own"][low_cost],
        loss_rent=params["loss_rent"][low_cost],
        import_cost=params["imp_cost"][low_cost],
        import_delay=params["import_delay"][low_cost],
        finance_premium=params["fin_prem"][low_cost],
        sale_ratio=params["sale_ratio"][high_cost],
        other_supply=params["other_supply"][high_cost],
        buy_budget=params["buy_budget"][1],
        max_buy_share=params["max_buy_share"][2],
        import_cap=1.5e6 if large_import_cap else params["import_cap"][2],
        requirement=4.2e6,
        capacity=1105510.0,
    )


def calibrate(series):
    surplus = series[series.cfs > series.req]
    stock = series.bought + series.carry
    cleared = ((stock - series.carry.shift(-1)) / np.maximum(stock - 5e5, 1)).where(stock > 5e5).clip(0, 1)
    return dict(
        log_ratio=series.log_ratio.values,
        price_support=float((surplus.bought / surplus.cfs).mean()),
        clearing=float(cleared.mean()),
        forecast_sd=0.04,
    )


class Shocks:
    def __init__(self, cal, paths, horizon, seed):
        g = np.random.default_rng(seed)
        self.log_ratio = cal["log_ratio"][g.integers(0, len(cal["log_ratio"]), (paths, horizon))] + 0.05 * g.standard_normal((paths, horizon))
        self.forecast = cal["forecast_sd"] * g.standard_normal((paths, horizon))
        self.paths, self.horizon = paths, horizon


def step(stock, harvest, forecast, requirement, capacity, policy, par):
    target, support, clearing = policy
    bought = np.minimum(np.maximum(0, target - stock) + support * harvest * (forecast > requirement), par["max_buy_share"] * harvest)
    held = stock + bought
    own, rented = np.minimum(held, capacity), np.maximum(0, held - capacity)
    loss = par["loss_own"] * own + par["loss_rent"] * rented
    holding = par["hold"] * own + (par["hold" ] + par["rent_extra"]) * rented
    after_loss = held - loss
    need = np.maximum(0, requirement - (harvest - bought) - par["other_supply"] * requirement)
    released = np.minimum(after_loss, need)
    cleared = clearing * np.maximum(0, after_loss - released - target)
    remaining = need - released
    imports = np.minimum(remaining, par["import_cap"])
    unmet = np.maximum(0, remaining - (1 - par["import_delay"]) * imports)
    next_stock = after_loss - released - cleared
    financing = par["finance_premium"] * np.maximum(0, bought - par["buy_budget"])
    cash = bought - par["sale_ratio"] * (released + cleared) + holding + par["import_cost"] * imports + financing
    cost = cash - par["sale_ratio"] * (next_stock - stock)
    return dict(stock=next_stock, cost=cost, cash=cash, unmet=unmet, bought=bought, rented=rented, imports=imports)


def simulate(policy, par, shocks):
    stock = np.full(shocks.paths, float(policy[0]))
    costs, cash, unmet = [], [], []
    for t in range(shocks.horizon):
        harvest = par["requirement"] * np.exp(shocks.log_ratio[:, t])
        forecast = harvest * np.exp(shocks.forecast[:, t])
        out = step(stock, harvest, forecast, par["requirement"], par["capacity"], policy, par)
        stock = out["stock"]
        costs.append(out["cost"]); cash.append(out["cash"]); unmet.append(out["unmet"])
    costs, cash, unmet = np.array(costs), np.array(cash), np.array(unmet)
    flat = np.sort(costs.ravel())
    worst = flat[-max(1, int(0.05 * flat.size)):]
    return dict(mean_cost=costs.mean(), tail_cost=worst.mean(), peak_cash=np.percentile(cash.max(axis=0), 95),
                shortage_prob=(unmet > 0).mean(), unmet_kt=unmet.mean() / 1e3)


def cheapest(target_prob, par, shocks):
    best = None
    for reserve in GRID_R:
        result = simulate((reserve, 0.0, 0.0), par, shocks)
        if result["shortage_prob"] <= target_prob and (best is None or result["mean_cost"] < best[1]["mean_cost"]):
            best = ((reserve, 0.0, 0.0), result)
    return best


def replay(policy, series, par):
    stock = np.array([float(series.carry.iloc[0])])
    rows = []
    for _, y in series.iterrows():
        out = step(stock, np.array([y.cfs]), np.array([y.cfs]), y.req, y.cap, policy, par)
        stock = out["stock"]
        rows.append(dict(year=int(y.year), model_bought=out["bought"][0], actual_bought=y.bought,
                         model_end_stock=stock[0], imports=out["imports"][0], unmet=out["unmet"][0]))
    return pd.DataFrame(rows)


def run():
    pd.set_option("display.width", 230)
    params = load_parameters()
    series = load_series()
    par = build_case(params)
    cal = calibrate(series)
    shocks = Shocks(cal, PATHS, HORIZON, SEED)

    policies = {
        "Fixed 300k": (3e5, 0.0, cal["clearing"]),
        "Fixed 500k": (5e5, 0.0, cal["clearing"]),
        "Fixed 1.0m": (1e6, 0.0, cal["clearing"]),
        "FRA-like": (5e5, cal["price_support"], cal["clearing"]),
    }
    comparison = pd.DataFrame([dict(policy=name, reserve=p[0], **simulate(p, par, shocks)) for name, p in policies.items()])

    curve_rows = []
    for target in TARGETS:
        best = cheapest(target, par, shocks)
        curve_rows.append(dict(target=target, reserve=best[0][0], **best[1]) if best else dict(target=target, reserve=np.nan))
    curve = pd.DataFrame(curve_rows)

    cases = [("base", "base", False, 1.0), ("cheap parameters", "cheap", False, 1.0), ("costly parameters", "costly", False, 1.0),
             ("import cap 1.5m t", "base", True, 1.0), ("holding cost x0.5", "base", False, 0.5), ("holding cost x2", "base", False, 2.0)]
    sensitivity_rows = []
    for label, case, large, scale in cases:
        best = cheapest(0.05, build_case(params, case, large, scale), shocks)
        sensitivity_rows.append(dict(setting=label, reserve=best[0][0] if best else np.nan, mean_cost=best[1]["mean_cost"] if best else np.nan))
    for label, share in (("other supply low", params["other_supply"][0]), ("other supply high", params["other_supply"][2])):
        best = cheapest(0.05, {**par, "other_supply": share}, shocks)
        sensitivity_rows.append(dict(setting=label, reserve=best[0][0] if best else np.nan, mean_cost=best[1]["mean_cost"] if best else np.nan))
    sensitivity = pd.DataFrame(sensitivity_rows)

    chosen = cheapest(0.05, par, shocks)
    history = replay(chosen[0], series, par) if chosen else pd.DataFrame()

    print("Other supply share (low, base, high):", tuple(round(v, 3) for v in params["other_supply"]))
    print(f"Price support share {cal['price_support']:.2f}, clearing share {cal['clearing']:.2f}")
    print(f"Costs are in tonnes at floor price; multiply by {FLOOR_PRICE_PER_TONNE:,} for kwacha")
    print("\nPolicies\n", comparison.round(3).to_string(index=False))
    print("\nCheapest policy by reliability target\n", curve.round(3).to_string(index=False))
    print("\nSensitivity at a 5% target\n", sensitivity.round(3).to_string(index=False))
    print("\nReplay\n", history.round(0).to_string(index=False))

    with pd.ExcelWriter(RESULTS) as writer:
        comparison.to_excel(writer, sheet_name="Policy comparison", index=False)
        curve.to_excel(writer, sheet_name="Cost vs reliability", index=False)
        sensitivity.to_excel(writer, sheet_name="Sensitivity", index=False)
        history.to_excel(writer, sheet_name="Replay", index=False)


if __name__ == "__main__":
    run()
