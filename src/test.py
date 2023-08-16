x = 'this is prefix Q: sdomhr A: this is answer Q: this is new q A: this is answer Q: this is suffix'
start = x.find('Q: ')
end = x.rfind('Q: ')
print(start)
print(end)
print(x[start:end])