def changeme(myvar):
    print('In funciotn, after change:', myvar)
    myvar = 50
    print('Outside function:', myvar)
    
myvar = 20
changeme(myvar)
print('Outside function:', myvar)
