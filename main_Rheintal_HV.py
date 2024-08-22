"""
Rheintal Multi-Hub main script
"""

import os
import shutil
from time import sleep
from pathlib import Path
from collections import OrderedDict
from ehubX.preprocessing.timer import time_it

from pyomo.core import Model, Objective, Var, Constraint

from ehubX.preprocessing.clustering_settings import ClusteringSettings
from ehubX.preprocessing.normalize_data import PerUnitFactors
from ehubX.run.run_model import GurobiSolverSettings, solve_model
from ehubX.postprocessing.result_plots import plot_pareto_front
from ehubX.postprocessing import save_model_results
from ehubX.run import define_general_model, prepare_inputs

# SINGLE OBJECTIVES
DO_OBJ_COST = True
DO_OBJ_CO2 = False
DO_OBJ_AUTARKY = False

# MULTI OBJECTIVES
NR_PARETO_POINTS = 4
DO_OBJ_COST_VS_CO2 = False
DO_OBJ_COST_VS_AUTARKY = False
DO_OBJ_AUTARKY_VS_CO2 = False

cur_script_path = os.path.abspath(os.path.dirname(__file__))
YAML_INPUTS = cur_script_path / Path("./inputs")
RESULT_FOLDER = cur_script_path / Path("./results")
MODEL_PPRINT = RESULT_FOLDER / Path("model_pprint.json")
DO_PPRINT = False

# You can add any Gurobi parameters here and it should be added to the solver settings
# Please use only valid Gurobi parameters listed under here https://www.gurobi.com/documentation/9.5/refman/parameters.html
solver_settings = GurobiSolverSettings(
    MIPGap=0.1,  # default: 0.0001 0.005 = 0.5% 0.01 = 1%
    MIPFocus=2,  # default = 0. Finding feasible sol. quicker: 1. Higher focus on optimality of solution: 2, best objective bound moves very slowly: 3
    NumericFocus=0,  # default: 0. max: 3
    Threads=8,  # default: 0 -> automatic
    # -1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex.
    Method=3,
    # off (0), conservative (1), or aggressive (2) -> the higher the tighter the model to solve
    Presolve=2,
    # default -1.Turning off scaling (ScaleFlag=0) can sometimes produce smaller constraint violations. Choosing a different scaling option can sometimes improve performance for particularly numerically difficult models (upt to 3).)
    ScaleFlag=3,
    SubMIPCuts=2,  # default -1, 0 to disable these cuts, 1 for moderate cut generation, or 2 for aggressive cut generation
    LogFile="Rheintal_Example_Log",
    OutputFlag=1,  # Enables or disables solver output
)


def project_specific_constraints(model: Model):

    #Rule to couple 25% of solar PV peak installed power (kW) to battery storage capacity (kWh)
    def capPVBatCoupled_rule_2050(model, s, h, c_coupled):
        if c_coupled == "T_S_Stor_Elec_Battery_ElBdg_2050":
            return model.V_CapTech[s, h, c_coupled] >= sum(model.V_CapTech[s, h, c_pv] for c_pv in model.S_SolarTech if "T_C_Sol_Solar_Pv_Roof_2050" in str(c_pv)) * 0.25 * 0.2 # 25% Capacity coupling Battery to PV (kWh/kW). 20% Solar efficiency.
        else:
            return Constraint.Skip

    model.C_CapPVBatCoupled_2050 = Constraint(
        model.Stage, model.Hub, model.Tech, rule=capPVBatCoupled_rule_2050)

    def chp_sum_rule(model, s, h, c_wood, c_gas):
        if c_wood == "T_C_Conv_Elec_Wood_El47_Chp" and c_gas == "T_C_Conv_Elec_Ch4_El47_Chp":
            return model.V_CapTech[s, h, c_wood] + model.V_CapTech[s, h, c_gas] <= 55 #PerUnitFactors: energy=1_000_000
        else:
            return Constraint.Skip

    model.C_CapChpSum = Constraint(
        model.Stage, model.Hub, model.Tech, model.Tech, rule=chp_sum_rule)

#    def gevag_turbine_rule(model, s, h, c_ct1, c_ct2):
#        if c_ct1 == "T_C_Conv_Elec_Th400_El13_ThPhII_Cond1" and c_ct2 == "T_C_Conv_Elec_Th400_El13_ThLt_Cond2" and h == "H4":
#            return model.V_CapTech[s, h, c_ct1] + model.V_CapTech[s, h, c_ct2] >= 0.012 #PerUnitFactors: energy=1_000_000. 12_000 kW -> 0.012 GW
#        else:
#            return Constraint.Skip
#
#    model.C_GevagTurbine = Constraint(
#        model.Stage, model.Hub, model.Tech, model.Tech, rule=gevag_turbine_rule)

    def maxWood_rule(model, s, ei):
        if ei == "Wood":
            return sum(model.V_EImp[s, h, ei, t] * model.P_HoursPerTS[s, t]
                       for t in model.TS for h in model.Hub
                       if (s, h, ei) in model.S_ImpAllowed) <= 500 # Limit wood import to Rheintal to 500 GWh/a. PerUnitFactors: energy=1_000_000
        else:
            return Constraint.Skip
    model.C_MaxWoodAnnual = Constraint(model.Stage, model.EC, rule=maxWood_rule)

    pass

@time_it
def run_single_objective_optimization(model: Model, pyo_obj: Objective, result_filename: str):
    """
    Does run single objective optimization for minimizing costs.
    Writes out all variables in a table format.
    """
    pyo_obj.activate()
    #model.write('model_Rheintal_Sz1.lp', io_options={"symbolic_solver_labels": True})
    solve_model(model, solver_setup=solver_settings.create_setup)
    pyo_obj.deactivate()
    save_model_results.save_results(model, path=RESULT_FOLDER, filename=result_filename,
                                    format=save_model_results.ResultFormat.TABLE_CSV, only_non_zero_results=True)


@time_it
def run_multi_objective_optimization(model: Model, nr_of_pareto_points: int, obj1: Objective, obj2: Objective, bounded_var_obj1: Var, var_obj2):
    """
    Runs a multiobjective optimization cost versus co2.
    Writes out full set of variables per pareto point and an summary csv table
    with all the overall variable values of all pareto points.

    :param model: pyomo model to solve, without objective functions defined
    :type model: Model
    :param obj1: first objective to optimize
    :param obj2: second objective to optimize
    :param bounded_var_obj1: variable to be queried for first dimension of pareto-point, used for Y-Constraint calculation
    :param var_obj2: variable for 2nd objective to use for pareto front
    """
    result_folder = RESULT_FOLDER / Path(f"{obj1.name}_{obj2.name}")
    shutil.rmtree(result_folder, ignore_errors=True)
    sleep(1)  # without the sleep re-creating the folder just deleted fails...
    os.mkdir(result_folder)

    overall_vars_per_pp = run_multiobjective(
        model,
        obj1=obj1,
        obj2=obj2,
        obj1_var=bounded_var_obj1,
        result_folder=result_folder,
        nr_of_pareto_points=nr_of_pareto_points,
        save_solver_json_res=None,
        only_non_zero_results=True,
        solver_settings=solver_settings,
    )

    var_obj1_per_pp: OrderedDict = overall_vars_per_pp[bounded_var_obj1.name].to_dict(
        into=OrderedDict)
    var_obj2_per_pp: OrderedDict = overall_vars_per_pp[var_obj2.name].to_dict(
        into=OrderedDict)
    plot_pareto_front(var_obj1_per_pp, var_obj2_per_pp,
                      result_folder / Path("pareto_front.jpg"))
    return overall_vars_per_pp


if __name__ == "__main__":
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)

    result_filename = "results_min_costs"

    data = prepare_inputs.prepare_inputs(
        YAML_INPUTS,
        RESULT_FOLDER=RESULT_FOLDER,
        result_filename=result_filename,
        per_unit_factors=PerUnitFactors(cost=1000, energy=1_000_000, co2=1000),
        clustering_settings=ClusteringSettings(
            nr_of_typical_periods_ts=100, ts_max_fraction=0.3),
        nr_of_full_horizon_ts=None
    )

    model = define_general_model.generate_model(data, primaryenergy_enabled=False, autarky_enabled=False)
    project_specific_constraints(model)

    if DO_OBJ_COST:
        run_single_objective_optimization(
            model, model.O_TotalCost, "results_min_costs")
    if DO_OBJ_CO2:
        run_single_objective_optimization(
            model, model.O_TotalCarbon, "results_min_co2")
    if DO_OBJ_AUTARKY:
        run_single_objective_optimization(
            model, model.O_Autarky, "results_max_autarky")

    if DO_OBJ_COST_VS_CO2:
        run_multi_objective_optimization(
            model, nr_of_pareto_points=NR_PARETO_POINTS, obj1=model.O_TotalCost, obj2=model.O_TotalCarbon, bounded_var_obj1=model.V_TotalCost, var_obj2=model.V_TotalCarbon2
        )
    if DO_OBJ_COST_VS_AUTARKY:
        run_multi_objective_optimization(
            model, nr_of_pareto_points=NR_PARETO_POINTS, obj1=model.O_TotalCost, obj2=model.O_Autarky, bounded_var_obj1=model.V_TotalCost, var_obj2=model.V_AutarkySystem
        )
    if DO_OBJ_AUTARKY_VS_CO2:
        run_multi_objective_optimization(
            model, nr_of_pareto_points=NR_PARETO_POINTS, obj1=model.O_Autarky, obj2=model.O_TotalCarbon, bounded_var_obj1=model.V_AutarkySystem, var_obj2=model.V_TotalCarbon2
        )
