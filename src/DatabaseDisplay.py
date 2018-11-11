import pickle

#==================

pickle_in = open("DB.pickle", "rb")
Database = pickle.load(pickle_in)

print(Database)


print("Press any key to continue")
input()

