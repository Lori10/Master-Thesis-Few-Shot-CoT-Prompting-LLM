from typing import TypeVar, Generic, List

T = TypeVar('T')

class DataRepository(Generic[T]):
    def __init__(self):
        self.data = []

    def add_data(self, item: T) -> None:
        self.data.append(item)

    def remove_data(self, item: int) -> None:
        self.data.remove(item)

    def get_all_data(self) -> List[str]:
        return self.data

repo = DataRepository[int]()
repo.add_data(10)
repo.add_data(20)
repo.add_data(30)
print(repo.get_all_data()) # Output: [10, 20, 30]
repo.remove_data(20)
print(repo.get_all_data()) # Output: [10, 30]
