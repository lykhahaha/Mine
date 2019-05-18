import unittest

class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

def remove_duplicates(head):
    node = head

    values = [node.data]
    while node.next:
        if node.next.data in values:
            node.next = node.next.next
        else:
            values.append(node.next.data)
            node = node.next

    return head

class Test(unittest.TestCase):
    def test_remove_duplicates(self):
        head = Node(1, Node(3, Node(3, Node(1, Node(5)))))
        remove_duplicates(head)
        self.assertEqual(head.data, 1)
        self.assertEqual(head.next.data, 3)
        self.assertEqual(head.next.next.data, 5)
        self.assertEqual(head.next.next.next, None)

if __name__ == '__main__':
    unittest.main()