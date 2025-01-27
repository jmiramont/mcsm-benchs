"""mcms-benchs: A common framework for benchmarks of multi-component signal processing methods.

Copyright (C) 2024  Juan Manuel Miramont

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from abc import ABC, abstractmethod


class MethodTemplate(ABC):
    """An abstract class for new methods to add in a benchmark."""

    @abstractmethod
    def __init__(self): ...

    @abstractmethod
    def method(self): ...

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, task):
        self._task = task

    def get_parameters(self):
        return (((), {}),)
