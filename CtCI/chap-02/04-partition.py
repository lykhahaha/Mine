class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next
    def __str__(self):
        string = str(self.data)
        while self.next:
            string = ', '.join([string, str(self.next.data)])

def partition(head, pivot):
    lead, tail = None, None
    count_pivot = 0

    node = head

    while node.next:
        if node.data < pivot:
            lead = node
            lead = lead.next