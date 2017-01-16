

# classes in the problem
output_dict = {
    "Cause-Effect(e1,e2)" : 0,
    "Cause-Effect(e2,e1)" : 1,
    "Instrument-Agency(e1,e2)" : 2,
    "Instrument-Agency(e2,e1)" : 3,
    "Product-Producer(e1,e2)" : 4,
    "Product-Producer(e2,e1)" : 5,
    "Content-Container(e1,e2)" : 6,
    "Content-Container(e2,e1)" : 7,
    "Entity-Origin(e1,e2)" : 8,
    "Entity-Origin(e2,e1)" : 9,
    "Entity-Destination(e1,e2)" : 10,
    "Entity-Destination(e2,e1)" : 11,
    "Component-Whole(e1,e2)" : 12,
    "Component-Whole(e2,e1)" : 13,
    "Member-Collection(e1,e2)" : 14,
    "Member-Collection(e2,e1)" : 15,
    "Message-Topic(e1,e2)" : 16,
    "Message-Topic(e2,e1)" : 17,
    "Other" : 18
}

reverse_dict = {v: k for k, v in output_dict.items()}

