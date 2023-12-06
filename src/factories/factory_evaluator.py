from src.evaluators.evaluator_f1_macro_token_level import EvaluatorF1MacroTokenLevel
from src.evaluators.evaluator_acc_token_level import EvaluatorAccuracyTokenLevel

class EvaluatorFactory():
    """EvaluatorFactory contains wrappers to create various evaluators."""
    @staticmethod
    def create(args):
        if args.evaluator == 'f1-macro':
            return EvaluatorF1MacroTokenLevel()
        elif args.evaluator == 'token-acc':
            return EvaluatorAccuracyTokenLevel()
        else:
            raise ValueError('Unknown evaluator %s.' % args.evaluator)