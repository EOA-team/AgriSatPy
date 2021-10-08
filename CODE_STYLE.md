# CODE_STYLE

Establishing some common code styling rules can help increasing the readability and maintability of the code and is vital for collaborative development.

Here are some rules to follow:

## Variable Naming

Variable names shall be consice and verbose. Within the function bodies, there is no strict variable naming style, however,
we try to follow the [PEP8](https://realpython.com/python-pep8/) conventions whenever possible.

Therefore, for variable names (taken from PEP8):
Use a lowercase single letter, word, or words. Separate words with underscores to improve readability.

E.g.,
```python
i = 0
number = 1
string_variable = 'This is a string'
```

## Class Naming

Following [PEP8](https://realpython.com/python-pep8/) conventions, class names should start with a captital lette.
Class names consisting of multiple words should not be separated by an underscore, but follow the *camel case* convention:

E.g.,
```python
class Class(object):
   pass

class MyClass(Class):
   pass
```

## Function Headers

See below for an example how to style function headers. Please **always** use type declarations to indicate the datatypes required/returned.
Moreover, the in- and outputs of the function should be documented in the `reST` style.

```python
def fun(
   a: int,
   b: Union[int, float],
   c: Optional[str]=''
) -> int:
   """
   function description goes here

   :param a:
       description of a
   :param b:
       description of b
   :param c:
       description of c
   :returns:
       description of return value(s)
   """
   pass # function code...
```

## Comments
Please make inline comments to explain the code or why you opted for certain implementations. Also mention code sources in case
you took some fixes from, e.g., stackoverflow or similar portals. Please provide the URL and the date you accessed the page.

