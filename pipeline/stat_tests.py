"""
Actual statistical test implementations. Each function takes clean pandas
input (not raw ONS/Home Office API payloads — see the note in pipeline.py
about the adapter layer you'll need once you have live data) and returns
a TestResult.

These are standard, correctly-specified implementations of the tests
named in the framework discussion. What they are NOT: a substitute for a
statistician reviewing the actual model specification for a real claim
before publication. Encoding "run a White test" correctly doesn't make
the choice to run a White test on a particular dataset correct — that's
still a judgment call for whoever sets up the fetch/transform for a given
claim type.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white

from .types import Direction, TestResult


def _direction_from_coefficient(coef: float) -> Direction:
    return "positive" if coef >= 0 else "negative"


def run_linear_regression(
    x: pd.Series,
    y: pd.Series,
    x_name: str = "x",
    y_name: str = "y",
) -> TestResult:
    """
    Simple OLS: y ~ x. Returns the slope's sign as effect_direction and
    its p-value. Caveat included by default since a bivariate regression
    can't itself rule out confounds — that's the classifier's job to flag
    which confounds a fuller model needs to include as extra regressors.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]

    if len(x_arr) < 3:
        return TestResult(
            test_name="linear_regression",
            statistic=float("nan"),
            p_value=1.0,
            effect_direction=None,
            n_observations=len(x_arr),
            caveats=["Fewer than 3 valid observations — regression not meaningful."],
        )

    X = sm.add_constant(x_arr)
    model = sm.OLS(y_arr, X).fit()
    slope = model.params[1]
    p_value = model.pvalues[1]

    return TestResult(
        test_name="linear_regression",
        statistic=model.tvalues[1],
        p_value=float(p_value),
        effect_direction=_direction_from_coefficient(slope),
        effect_size=float(slope),
        n_observations=len(x_arr),
        caveats=[
            f"Bivariate regression of {y_name} on {x_name} only — does not "
            "control for confounds. Confirm any required control variables "
            "(from the classifier's confound list) were added as additional "
            "regressors before trusting this result alone.",
        ],
        raw={
            "r_squared": float(model.rsquared),
            "slope": float(slope),
            "slope_confidence_interval_95": [
                float(model.conf_int(alpha=0.05)[1][0]),
                float(model.conf_int(alpha=0.05)[1][1]),
            ],
        },
    )


def run_difference_in_differences(
    treatment_pre: pd.Series,
    treatment_post: pd.Series,
    control_pre: pd.Series,
    control_post: pd.Series,
) -> TestResult:
    """
    Classic 2x2 DiD via OLS with a treatment x post interaction term.
    Use this instead of comparing simple before/after averages, since it
    nets out any trend the control group shows on its own — the whole
    point of DiD.
    """
    y = np.concatenate([treatment_pre, treatment_post, control_pre, control_post])
    treatment = np.concatenate([
        np.ones(len(treatment_pre)), np.ones(len(treatment_post)),
        np.zeros(len(control_pre)), np.zeros(len(control_post)),
    ])
    post = np.concatenate([
        np.zeros(len(treatment_pre)), np.ones(len(treatment_post)),
        np.zeros(len(control_pre)), np.ones(len(control_post)),
    ])
    interaction = treatment * post

    X = sm.add_constant(np.column_stack([treatment, post, interaction]))
    model = sm.OLS(y, X).fit()

    did_coef = model.params[3]  # the interaction term IS the DiD estimate
    did_p = model.pvalues[3]

    return TestResult(
        test_name="difference_in_differences",
        statistic=model.tvalues[3],
        p_value=float(did_p),
        effect_direction=_direction_from_coefficient(did_coef),
        effect_size=float(did_coef),
        n_observations=len(y),
        caveats=[
            "DiD's validity rests on the 'parallel trends' assumption — that "
            "treatment and control groups would have moved together absent "
            "the intervention. This isn't tested automatically here; check "
            "pre-period trends visually or with a pre-trend test before "
            "trusting this result.",
        ],
        raw={"did_estimate": float(did_coef)},
    )


def run_chi_square(contingency_table: pd.DataFrame) -> TestResult:
    """
    Chi-square test of independence on a contingency table (e.g. rows =
    nationality/group, columns = offence categories or offend/no-offend).
    Note: chi-square tells you groups differ, not why — this is exactly
    where the age-structure confound needs to be controlled for
    separately (e.g. by standardizing rates by age band) before this
    test's result means what a headline would claim it means.
    """
    chi2, p, dof, expected = scipy_stats.chi2_contingency(contingency_table.values)

    n = contingency_table.values.sum()
    # Cramer's V as an effect size for chi-square
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else float("nan")

    return TestResult(
        test_name="chi_square",
        statistic=float(chi2),
        p_value=float(p),
        effect_direction=None,  # chi-square has no inherent direction
        effect_size=float(cramers_v),
        n_observations=int(n),
        caveats=[
            "Chi-square shows association, not direction or causation. "
            "Age-structure and other confounds (see classifier output) "
            "must be controlled for — e.g. via age-standardized rates — "
            "or this result will overstate the claim it's checking.",
        ],
        raw={"degrees_of_freedom": int(dof)},
    )


def run_white_test(x: pd.Series, y: pd.Series) -> TestResult:
    """
    White's test for heteroscedasticity, run on the residuals of a
    y ~ x regression. This doesn't test the claim itself — it tests
    whether the regression used to test the claim has a problem (non-
    constant error variance) that would make its p-values unreliable.
    Run this as a diagnostic alongside run_linear_regression, not instead
    of it.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]

    if len(x_arr) < 5:
        return TestResult(
            test_name="white_test_heteroscedasticity",
            statistic=float("nan"),
            p_value=1.0,
            effect_direction=None,
            n_observations=len(x_arr),
            caveats=["Fewer than 5 observations — White test not meaningful."],
        )

    X = sm.add_constant(x_arr)
    model = sm.OLS(y_arr, X).fit()
    lm_stat, lm_p, f_stat, f_p = het_white(model.resid, X)

    return TestResult(
        test_name="white_test_heteroscedasticity",
        statistic=float(lm_stat),
        p_value=float(lm_p),
        effect_direction=None,
        n_observations=len(x_arr),
        caveats=[
            "This tests the regression's error variance, not the claim "
            "itself. A significant result (p < 0.05) means the accompanying "
            "linear_regression's standard errors are unreliable — use "
            "heteroscedasticity-robust standard errors (e.g. HC3) and "
            "re-check significance before trusting that test's p-value.",
        ],
    )


def run_two_sample_t_test(group_a: pd.Series, group_b: pd.Series) -> TestResult:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]

    if len(a) < 2 or len(b) < 2:
        return TestResult(
            test_name="two_sample_t_test",
            statistic=float("nan"),
            p_value=1.0,
            effect_direction=None,
            caveats=["Insufficient observations in one or both groups."],
        )

    t_stat, p_value = scipy_stats.ttest_ind(a, b, equal_var=False)  # Welch's, safer default
    mean_diff = float(np.mean(a) - np.mean(b))

    return TestResult(
        test_name="two_sample_t_test",
        statistic=float(t_stat),
        p_value=float(p_value),
        effect_direction=_direction_from_coefficient(mean_diff),
        effect_size=mean_diff,
        n_observations=len(a) + len(b),
        caveats=["Welch's t-test used (does not assume equal variances)."],
    )


def run_time_series_correlation(series_a: pd.Series, series_b: pd.Series) -> TestResult:
    """
    Pearson correlation between two time series. Deliberately flagged
    with a strong caveat: two trending series will correlate even with
    no real relationship (spurious correlation via shared trend) — this
    test alone should never be the sole basis for a verdict on a causal
    or associative claim, only a first-pass screen.
    """
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]

    if len(a) < 3:
        return TestResult(
            test_name="time_series_correlation",
            statistic=float("nan"),
            p_value=1.0,
            effect_direction=None,
            caveats=["Fewer than 3 paired observations."],
        )

    r, p_value = scipy_stats.pearsonr(a, b)

    return TestResult(
        test_name="time_series_correlation",
        statistic=float(r),
        p_value=float(p_value),
        effect_direction=_direction_from_coefficient(r),
        effect_size=float(r),
        n_observations=len(a),
        caveats=[
            "Raw correlation between two time series is prone to spurious "
            "results when both series trend over time. This should be "
            "treated as a screening step, not sufficient evidence alone — "
            "pair with a DiD or regression that controls for trend.",
        ],
    )


TEST_REGISTRY = {
    "linear_regression": run_linear_regression,
    "difference_in_differences": run_difference_in_differences,
    "chi_square": run_chi_square,
    "white_test_heteroscedasticity": run_white_test,
    "two_sample_t_test": run_two_sample_t_test,
    "time_series_correlation": run_time_series_correlation,
}
