from classifier import RuleEmbeddingClassifier

clf = RuleEmbeddingClassifier()

if __name__ == "__main__":
    while True:
        q = input("User: ")
        print("Intent:", clf.predict(q))
