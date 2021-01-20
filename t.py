
vip_user = ["a","b","c","d","e","f","g"]
buyer = ["b","c","d","e","h","j"]

print(set(vip_user) - set(buyer)) # vip_user를 기준으로 차집합 연산 "-"
#set( ["a"
print(set(buyer) & set(vip_user)) # buyer 와 vip_user의 교집합 "&"
print(set(buyer) ^ set(vip_user)) # 대칭차집합 연산 


from collections import deque
a = [ 1,2,3,5 ]
deque_list = deque(a)
print(deque_list)



deque_list = deque( a )
deque_list.append(12) # 오른쪽 12입력
deque_list.appendleft(10) # 왼쪽으로 12 입력
print(deque_list)
deque_list.insert(1,30)
deque_list.rotate(2)
print(deque_list)