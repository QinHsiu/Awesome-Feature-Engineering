# some python tricks
# 1. for循环中的else条件
def for_else():
    numbers=[2,4,6,8,1]
    for num in numbers:
        if num%2==0:
            print(num)
            break
    else:
        # 只有在正常退出循环（而不是break）才会执行该语句
        print("No even numbers")

def list_split():
    my_list=[i for i in range(1,3)]
    # 将列表元素分别赋予不同变量
    one,two=my_list
    print(one,two)

# 求解列表中最大或最小的几个元素
import heapq
import random
def get_max():
    scores=[i for i in range(1,100)]
    random.shuffle(scores)
    print("最大的三个元素：",heapq.nlargest(3,scores))
    print("最小的三个元素：",heapq.nsmallest(3,scores))

# 将列表元素作为参数进行传递
def list_to_param():
    my_list=[1,2,3,4]
    print(my_list)
    print(*my_list)

    # 获取中间元素
    _,*element,_=my_list
    print(element)

# 使用枚举
from enum import Enum
class Status(Enum):
    NO_STATUS = -1
    NOT_STARTED = 0
    IN_PROGRESS = 1
    COMPLETED = 2
def get_enum():
    print(Status.IN_PROGRESS.name)  # IN_PROGRESS
    print(Status.COMPLETED.value)  # 2

# 合并字典
def concat_two_dict():
    first_dictionary = {'name': 'Fan', 'location': 'Guangzhou'}
    second_dictionary = {'name': 'Fan', 'surname': 'Xiao', 'location': 'Guangdong, Guangzhou'}
    # 方法一
    result = first_dictionary | second_dictionary
    # 方法二
    merged={**first_dictionary,**second_dictionary}

    print(result,merged)

# 将字符串转换为字符串列表
import ast

def string_to_list(string):
    return ast.literal_eval(string)

def get_string2list():
    string = "[1, 2, 3]"
    my_list = string_to_list(string)
    print(my_list)  # [1, 2, 3]
    string = "[[1, 2, 3],[4, 5, 6]]"
    my_list = string_to_list(string)
    print(my_list)  # [[1, 2, 3], [4, 5, 6]]

# is和===的区别是，is检查两个变量是不是指向同一个对象内存中；==比较两个对象的值
def test_is_and_equal():
    first_list = [1, 2, 3]
    second_list = [1, 2, 3]
    # 比较两个值
    print(first_list == second_list)  # True
    # 是否指向同一内存
    print(first_list is second_list)
    # False
    third_list = first_list
    print(third_list is first_list)

# 设置静态列表
def get_stastic_list():
    my_set = frozenset(['a', 'b', 'c', 'd'])
    my_set.add("a")

# 使用filter创建新对象
def use_filter():
    my_list = [1, 2, 3, 4]
    odd = filter(lambda x: x % 2 == 1, my_list)
    print(list(odd))  # [1, 3]
    print(my_list)  # [1, 2, 3, 4]

# 使用map创建一个新对象
def use_map():
    my_list = [1, 2, 3, 4]
    squared = map(lambda x: x ** 2, my_list)
    print(list(squared))  # [1, 4, 9, 16]
    print(my_list)


# 检查内存使用情况
import sys
def check_sys():
    print(sys.getsizeof("bitcoin"))  # 56

# 操作符重载
# 加法
class Expenses:
    def __init__(self, rent, groceries):
        self.rent = rent
        self.groceries = groceries

    def __add__(self, other):
        return Expenses(self.rent + other.rent,
                        self.groceries + other.groceries)

# 等于符号
class Journey:
    def __init__(self, location, destination, duration):
        self.location = location
        self.destination = destination
        self.duration = duration

    def __eq__(self, other):
        return ((self.location == other.location) and
                (self.destination == other.destination) and
                (self.duration == other.duration))

# 小于符号重载
class Game:
    def __init__(self, score):
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

# 打印函数
class Rectangle:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self):
        return repr('Rectangle with area=' + str(self.a * self.b))


def test_operation():
    april_expenses = Expenses(1000, 200)
    may_expenses = Expenses(1000, 300)
    total_expenses = april_expenses + may_expenses
    print(total_expenses.rent)  # 2000
    print(total_expenses.groceries)  # 500

    first = Game(1)
    second = Game(2)
    print(first < second)  # True

    first = Journey('Location A', 'Destination A', '30min')
    second = Journey('Location B', 'Destination B', '30min')
    print(first == second)

    print(Rectangle(3, 4))  # 'Rectangle with area=12'

# 使用slice截取元素
def use_slice():
    my_list = list(range(1,11))
    slicing = slice(-4, None)
    print(my_list[slicing])  # [7, 8, 9,10]
    print(my_list[-3])  # 8


def delete_elem():
    my_list = [1, 2, 3, 4]
    my_list.clear()
    print(my_list)  # []

    my_set = {1, 2, 3}
    my_set.clear()
    print(my_set)  # set()

    my_dict = {"a": 1, "b": 2}
    my_dict.clear()
    print(my_dict)  # {}

# 使用collection中的方法求解元素出现次数
def get_count():
    from collections import Counter
    result = Counter("Banana")
    print(result)  # Counter({'a': 3, 'n': 2, 'B': 1})
    my_list=[1, 2, 1, 3, 1, 4, 1, 5, 1, 6]
    result = Counter(my_list)
    print(result)  # Counter({1: 5, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})

    # 使用该方法按元素出现次数进行排序
    print(result.most_common())

    # 查找列表元素出现次数最高的元素
    print(max(set(my_list), key=my_list.count))  # a


# 使用迭代方法计算元素出现次数
def get_count_by_iter():
    from itertools import count
    my_vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
    current_counter = count()
    string = "This is just a sentence."
    for i in string:
        if i in my_vowels:
            print(f"Current vowel: {i}")
            print(f"Number of vowels found so far: {next(current_counter)}")



# 构建迭代器
class OddNumbers:
    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        x = self.a
        self.a += 2
        return x

def test_iter():
    odd_numbers_object = OddNumbers()
    iterator = iter(odd_numbers_object)
    print(next(iterator))  # 1
    print(next(iterator))  # 3
    print(next(iterator))  # 5


# 生成唯一标识id
def get_id():
    import uuid
    # 根据主机ID、序列号和当前时间生成UUID
    print(uuid.uuid1())  # 308490b6-afe4-11eb-95f7-0c4de9a0c5af
    # 生成一个随机UUID
    print(uuid.uuid4())  # 93bc700b-253e-4081-a358-24b60591076a


if __name__ == '__main__':
    use_slice()