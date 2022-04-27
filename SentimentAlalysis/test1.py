# pip install textblob
from textblob import TextBlob

fb1 = "The Food at Abc was awesome"
fb1 = "The Food at Abc was bad"
fb2= "The food is Abc was very good"

bob1 = TextBlob(fb1)
bob2 = TextBlob(fb2)

print(bob1.sentiment)
print(int((bob1.sentiment.polarity/2 )*10) )
print(bob1.sentences)
print(bob2.sentiment)

t1 = "Titanic is a great movie."
t2 = "Titanic is not a great movie."
t3 = "Titanic is a movie."

b1 = TextBlob(t1)
b2 = TextBlob(t2)
b3 = TextBlob(t3)

print(b1.sentiment)
print(b2.sentiment)
print(b3.sentiment)
