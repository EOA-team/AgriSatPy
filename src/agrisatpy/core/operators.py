'''
Base class defining map algebra operators
'''

from typing import List

class Operator:
    """
    Band operator supporting basic algebraic operations
    """
    operators: List[str] = ['+', '-', '*', '/', '**', '<', '>', '==', '<=', '>=']

    class BandMathError(Exception):
        pass

    @classmethod
    def check_operator(cls, operator: str) -> None:
        """
        Checks if the operator passed is valid

        :param operator:
            passed operator to evaluate
        """
        # check operator passed first
        if operator not in cls.operators:
            raise ValueError(f'Unknown operator "{operator}"')
    
    @classmethod
    def calc(cls):
        """
        Class method to be overwritten by inheriting class
        """
        pass