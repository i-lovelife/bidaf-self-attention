from typing import Dict, List

from overrides import overrides

from allennlp.training.metrics.metric import Metric

from external import evaluate_v2

@Metric.register("squad_v2")
class SquadMetricsV2(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed exact match and F1 score using the official SQuAD-v2
    evaluation script. Calculate statistics for question with no answer seprately
    """
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        self._total_no_em = 0.0
        self._total_no_f1 = 0.0
        self._no_count = 0
        self._total_yes_em = 0.0
        self._total_yes_f1 = 0.0
        self._yes_count = 0

    @overrides
    def __call__(self, best_span_string: str, answer_strings: List[str]) -> None:
        """
        Parameters
        ----------
        best_span_string : ``str``
            The answer string predicted by model
        answer_strings : ``List[str]``
            The golden span strings, may be several
        """
        no_answer = False
        if not answer_strings or not answer_strings[0]:
            answer_strings = ['']
            no_answer = True
        exact_match = max(evaluate_v2.compute_exact(answer, best_span_string) for answer in answer_strings)
        f1_score = max(evaluate_v2.compute_f1(answer, best_span_string) for answer in answer_strings)
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1

        if no_answer:
            self._total_no_em += exact_match
            self._total_no_f1 += f1_score
            self._no_count += 1
        else:
            self._total_yes_em += exact_match
            self._total_yes_f1 += f1_score
            self._yes_count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official SQuADV2 script
        over all inputs.
        """
        def ave_score(value, count):
            return value/count if count > 0 else 0
        total_em = ave_score(self._total_em, self._count)
        total_f1 = ave_score(self._total_f1, self._count)
        total_no_em = ave_score(self._total_no_em, self._no_count)
        total_no_f1 = ave_score(self._total_no_f1, self._no_count)
        total_yes_em = ave_score(self._total_yes_em, self._yes_count)
        total_yes_f1 = ave_score(self._total_yes_f1, self._yes_count)
        ret = {
                "em": total_em,
                "f1": total_f1,
                "no_em": total_no_em,
                "yes_em": total_yes_em
        }
        if reset:
            self.reset()
        return ret

    @overrides
    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        self._total_no_em = 0.0
        self._total_no_f1 = 0.0
        self._no_count = 0
        self._total_yes_em = 0.0
        self._total_yes_f1 = 0.0
        self._yes_count = 0

    def __str__(self):
        return f"SquadEmAndF1V2(em={self._total_em}, f1={self._total_f1}, \
        no_em={self._total_no_em}, no_f1={self._total_no_f1}, \
        yes_em={self._total_yes_em}, yes_f1={self._total_yes_f1})"
