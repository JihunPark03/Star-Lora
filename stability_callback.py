import csv
import os

import numpy as np
from transformers import TrainerCallback


class StabilityAwareCallback(TrainerCallback):
    """
    Logs stable importance scores of LoRA parameters.

    Raw importance:
        I = mean(abs(weight * gradient))

    Stable importance:
        EMA(I) / (sqrt(Variance(I)) + eps)
    """

    def __init__(self, output_dir, beta=0.9, eps=1e-8):
        self.output_dir = output_dir
        self.beta = beta
        self.eps = eps

        self.ema = {}
        self.var = {}

        os.makedirs(output_dir, exist_ok=True)

        self.log_path = os.path.join(
            output_dir,
            "stability_scores.csv",
        )

        self._initialize_csv()

    def _initialize_csv(self):
        with open(self.log_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "step",
                    "parameter_name",
                    "raw_importance",
                    "stable_importance",
                ]
            )

    def on_pre_optimizer_step(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs,
    ):
        if model is None:
            return

        rows = []

        for name, parameter in model.named_parameters():
            if not self._is_lora_parameter(name, parameter):
                continue

            if parameter.grad is None:
                continue

            raw_importance = self._compute_raw_importance(parameter)
            stable_importance = self._update_stable_importance(
                name,
                raw_importance,
            )

            rows.append(
                [
                    state.global_step,
                    name,
                    raw_importance,
                    stable_importance,
                ]
            )

        self._write_rows(rows)

    def _is_lora_parameter(self, name, parameter):
        if not parameter.requires_grad:
            return False

        return "lora" in name.lower()

    def _compute_raw_importance(self, parameter):
        score = parameter.data * parameter.grad.data
        score = score.detach().abs().mean().cpu().item()

        return float(score)

    def _update_stable_importance(self, name, importance):
        if name not in self.ema:
            self.ema[name] = importance
            self.var[name] = 0.0
        else:
            old_ema = self.ema[name]

            self.ema[name] = (
                self.beta * old_ema
                + (1.0 - self.beta) * importance
            )

            self.var[name] = (
                self.beta * self.var[name]
                + (1.0 - self.beta) * ((importance - old_ema) ** 2)
            )

        stable_importance = self.ema[name] / (
            np.sqrt(self.var[name]) + self.eps
        )

        return float(stable_importance)

    def _write_rows(self, rows):
        if len(rows) == 0:
            return

        with open(self.log_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(rows)


class AdaLoraAllocationCallback(TrainerCallback):
    """
    Delegates adaptive rank allocation to PEFT's AdaLoRA implementation.
    """

    def on_optimizer_step(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs,
    ):
        if model is None:
            return

        base_model = getattr(model, "base_model", model)

        if hasattr(base_model, "update_and_allocate"):
            base_model.update_and_allocate(state.global_step)
