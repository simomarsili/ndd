# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc.
# and PyData Development Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
"""Appender class from pandas.util._decorators ."""
from textwrap import dedent
from typing import Any, Callable, Optional, TypeVar

FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)


class Appender:  # pylint: disable=too-few-public-methods
    """
    A function decorator that will append an addendum to the docstring
    of the target function.
    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).
    Usage: construct a docstring.Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.
    add_copyright = Appender("Copyright (c) 2009", join='\n')
    @add_copyright
    def my_dog(has='fleas'):
        "This docstring will have a copyright below"
        pass
    """

    def __init__(self,
                 addendum: Optional[str],
                 join: str = '',
                 indents: int = 0):
        if indents > 0:
            # type: Optional[str]
            self.addendum = indent(addendum, indents=indents)
        else:
            self.addendum = addendum
        self.join = join

    def __call__(self, func: F) -> F:
        func.__doc__ = func.__doc__ if func.__doc__ else ''
        self.addendum = self.addendum if self.addendum else ''
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = dedent(self.join.join(docitems))
        return func


def indent(text: Optional[str], indents: int = 1) -> str:
    """Indent docstring text"""
    if not text or not isinstance(text, str):
        return ''
    jointext = ''.join(['\n'] + ['    '] * indents)
    return jointext.join(text.split('\n'))
