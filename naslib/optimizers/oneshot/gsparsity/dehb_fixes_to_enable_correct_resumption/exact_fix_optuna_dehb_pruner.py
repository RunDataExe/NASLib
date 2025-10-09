from __future__ import annotations

import math

import numpy as np

from optuna import logging
from optuna.pruners import HyperbandPruner
from optuna.pruners import SuccessiveHalvingPruner
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

_logger = logging.get_logger(__name__)


def _clean_trials(
    study: Study, trials: list[FrozenTrial], current_trial: FrozenTrial
) -> list[FrozenTrial]:
    """
    Return only the trials that are after the last invalid trial.
    This excludes the current running trial from being considered 'invalid'.
    """
    # Find the last trial that is in a FAIL or RUNNING state.
    last_invalid_trial = None
    for t in reversed(trials):
        # Ignore the current trial itself, as it's always RUNNING at this stage.
        if t.number == current_trial.number:
            continue

        if t.state in [TrialState.FAIL, TrialState.RUNNING]:
            last_invalid_trial = t
            break

    # If no such trial exists, the history is clean.
    if last_invalid_trial is None:
        return trials

    _logger.info(
        f"Using proactively cleaned history due to last invalid trial: "
        f"Trial {last_invalid_trial.number} (State: {last_invalid_trial.state})."
        f"Sampler only uses trials after this point."
    )

    unique_trials_list = [t for t in trials if t.number > last_invalid_trial.number]

    return unique_trials_list


class DEHBPruner(HyperbandPruner):
    def __init__(
        self,
        min_resource: int = 1,
        max_resource: str | int = "auto",
        reduction_factor: int = 3,
        bootstrap_count: int = 0,
    ) -> None:
        super().__init__(min_resource, max_resource, reduction_factor, bootstrap_count)

        self._budget_candidates: list[int] | None = None
        self._n_existing_trials: int | None = (
            None  # Cache for the number of pre-existing trials
        )

    def get_run_local_trial_index(self, study: Study, trial: FrozenTrial) -> int:
        """
        Calculates a trial's index local to its execution run.
        This handles resumed studies by treating the first trial of a new run as index 0.
        """
        # Determine and cache the number of pre-existing trials for this execution run.
        # This is done only once per pruner instance (i.e., per script execution).
        if self._n_existing_trials is None:
            # Get all trials from the storage.
            all_trials = study.get_trials(deepcopy=False)
            # The number of pre-existing trials is the count of all trials that are NOT the current one.
            # This correctly handles the case where the script is restarted.
            self._n_existing_trials = len(
                [t for t in all_trials if t.number != trial.number]
            )
            _logger.info(
                f"New execution run detected. Found {self._n_existing_trials} pre-existing trials."
            )

        # The local index is the trial's absolute number minus the number of trials from previous runs.
        return trial.number - self._n_existing_trials

    def prune(self, study: Study, trial: FrozenTrial) -> bool:
        if len(self._pruners) == 0:
            self._try_initialization(study)
            if len(self._pruners) == 0:
                return False

        if self._get_iteration_id(study, trial) == 0:
            if self._prune_for_initial_iteration(study, trial):
                return True
            return False

        bracket_id = self._get_bracket_id_after_init(study, trial)
        _logger.debug("{}th bracket is selected".format(bracket_id))
        bracket_study = self._create_bracket_study(study, bracket_id)
        return self._pruners[bracket_id].prune(bracket_study, trial)

    def _prune_for_initial_iteration(self, study: Study, trial: FrozenTrial) -> bool:
        if self._budget_candidates is None:
            self._create_budget_candidates(study)
        assert self._budget_candidates is not None

        assert self._get_iteration_id(study, trial) == 0

        bracket_id = self._get_bracket_id_after_init(study, trial)
        budget_id = self._get_budget_id(study, trial, bracket_id)
        budget = self._budget_candidates[budget_id]
        last_step = trial.last_step
        if last_step is None:
            return False
        if last_step < budget:
            return False

        return True

    def _create_budget_candidates(self, study: Study) -> None:
        assert self._n_brackets is not None
        successive_halving_pruner = self._pruners[0]
        assert isinstance(successive_halving_pruner, SuccessiveHalvingPruner)
        min_resource = successive_halving_pruner._min_resource
        reduction_factor = successive_halving_pruner._reduction_factor

        budget_candidates = []
        for i in range(self._n_brackets):
            budget_candidates.append(min_resource * reduction_factor**i)
        self._budget_candidates = budget_candidates

    def _get_n_trials_in_first_dehb_iteration(self, study: Study) -> int:
        if self._n_brackets is None:
            return 2**32 - 1  # Return a large number.

        s_max = self._n_brackets - 1

        successive_halving_pruner = self._pruners[0]
        assert isinstance(successive_halving_pruner, SuccessiveHalvingPruner)

        reduction_factor = successive_halving_pruner._reduction_factor

        n = 0
        for i in range(s_max + 1):
            for j in range(i, s_max + 1):
                n += reduction_factor ** (s_max - j)
        return n

    def _get_iteration_id(self, study: Study, trial: FrozenTrial) -> int:
        n_trials_per_iteration = self._get_n_trials_in_first_dehb_iteration(study)
        normalized_index = self.get_run_local_trial_index(study, trial)
        return normalized_index // n_trials_per_iteration

    def _get_bracket_id_after_init(self, study: Study, trial: FrozenTrial) -> int:
        assert isinstance(self._n_brackets, int)
        s_max = self._n_brackets - 1

        successive_halving_pruner = self._pruners[0]
        assert isinstance(successive_halving_pruner, SuccessiveHalvingPruner)

        reduction_factor = successive_halving_pruner._reduction_factor

        n_trials_for_each_bracket = []
        for i in range(s_max + 1):
            n = 0
            for j in range(i, s_max + 1):
                n += reduction_factor ** (s_max - j)
            n_trials_for_each_bracket.append(n)

        n_trials_per_iteration = self._get_n_trials_in_first_dehb_iteration(study)
        normalized_index = self.get_run_local_trial_index(study, trial)
        trial_number_in_iteration = normalized_index % n_trials_per_iteration
        for i in range(s_max + 1):
            if trial_number_in_iteration < n_trials_for_each_bracket[i]:
                return i
            trial_number_in_iteration -= n_trials_for_each_bracket[i]

        assert False, "This line should never be reached."

    def _get_current_budget(self, study: Study, trial: FrozenTrial) -> int:
        """Calculates the budget for the current trial based on its bracket."""
        # Ensure the pruner is initialized before proceeding.
        if len(self._pruners) == 0:
            self._try_initialization(study)
            if len(self._pruners) == 0:
                # This can happen if there are not enough completed trials yet.
                # Return a default or minimum resource.
                return self._min_resource

        if self._budget_candidates is None:
            self._create_budget_candidates(study)
        assert self._budget_candidates is not None

        bracket_id = self._get_bracket_id_after_init(study, trial)
        budget_id = self._get_budget_id(study, trial, bracket_id)
        return self._budget_candidates[budget_id]

    def _get_budget_id(self, study: Study, trial: FrozenTrial, bracket_id: int) -> int:
        budget_candidates = self._budget_candidates
        assert budget_candidates is not None
        assert isinstance(self._n_brackets, int)
        s_max = self._n_brackets - 1

        successive_halving_pruner = self._pruners[0]
        assert isinstance(successive_halving_pruner, SuccessiveHalvingPruner)

        reduction_factor = successive_halving_pruner._reduction_factor

        if trial.state != TrialState.RUNNING:
            last_step = trial.last_step
            assert last_step is not None

            successive_halving_pruner = self._pruners[0]
            assert isinstance(successive_halving_pruner, SuccessiveHalvingPruner)
            reduction_factor = successive_halving_pruner._reduction_factor

            difference = [
                abs(math.log(x / last_step, reduction_factor))
                for x in budget_candidates
            ]
            return np.argmin(difference)

        n_trials_for_each_bracket = []
        for i in range(s_max + 1):
            n = 0
            for j in range(i, s_max + 1):
                n += reduction_factor ** (s_max - j)
            n_trials_for_each_bracket.append(n)

        n_trials_in_the_bracket = []
        for j in range(bracket_id, s_max + 1):
            n = reduction_factor ** (s_max - j)
            n_trials_in_the_bracket.append(n)

        n_trials_per_iteration = self._get_n_trials_in_first_dehb_iteration(study)
        normalized_index = self.get_run_local_trial_index(study, trial)
        trial_number_in_iteration = normalized_index % n_trials_per_iteration
        for i in range(0, bracket_id):
            trial_number_in_iteration -= n_trials_for_each_bracket[i]

        for j in range(bracket_id, s_max + 1):
            if trial_number_in_iteration < n_trials_in_the_bracket[j - bracket_id]:
                return j
            trial_number_in_iteration -= n_trials_in_the_bracket[j - bracket_id]

        assert False, "This line should never be reached."
