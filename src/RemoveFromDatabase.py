import pickle

#=================

# Call Database for reading
pickle_in = open("DB.pickle", "rb")
Database = pickle.load(pickle_in)

print("Who do you want to remove?")
name = input()

pickle_out = open("DB.pickle","wb")

# Delete Individual from the Database
del Database[str(name)]

pickle.dump(Database, pickle_out)
pickle_out.close()
