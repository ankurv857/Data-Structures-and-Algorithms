#author @ankurverma
#Data Structures & Algorithms class implementation

class Solution:
    def fib(self, N: int) -> int:
        series = [None]*(N+1)
        for i in range(N+1):
            if i == 0:
                series[i] = 0
            elif i ==1:
                series[i] = 1
            else:
                series[i] = series[i-1] + series[i-2]
        return series[N]

print(Solution().fib(10))

class Solution:
    def pancake(self, arr):
        answer = []
        value_to_sort = len(arr)
        while value_to_sort > 0:
            index = arr.index(value_to_sort)
            print('value_to_sort', value_to_sort,'index', index) ; exit()

    def flip(self, arr, len_arr):
        arr_s1 = arr[:len_arr]
        arr_s2 = arr[len_arr:]
        arr_ = [None]*len(arr_s1)
        for i in range(len(arr_s1)):
            arr_[i] = arr_s1[len(arr_s1) - i - 1]
        return arr_ + arr_s2

print(Solution().pancake([3,2,4,1]))

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _len = len(nums)
        for i in range(_len):
            for j in range(i+1, _len):
                if (nums[i] + nums[j] == target):
                    return [i] + [j]

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for index, num in enumerate(nums):
            _sum = target - num
            if _sum in dic:
                return [dic[_sum], index]
            else:
                dic[num] = index

class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        ans = []
        l = len(nums)
        for a in range(l):
            for b in range(a+1, l):
                for c in range(b+1, l):
                    for d in range(c+1, l):
                        if (nums[a] +nums[b] + nums[c]+nums[d] == target):
                            iter_ = [nums[a]] + [nums[b]] + [nums[c]]+[nums[d]]
                            iter_ = sorted(iter_)
                            ans.append(iter_)
        dedup = []
        for elem in ans:
            if elem not in dedup:
                dedup.append(elem)
        ans = dedup
        return ans
                            
        
def threeSum(self, nums):
        nums.sort()
        dic = collections.defaultdict(int)
        for n in nums: dic[n] += 1
        res = set()
        
        for i, n in enumerate(nums[:-2]):
            dic[n] -= 1
            if dic[n] == 0: del dic[n]
            rest = dic.keys()
            for a in rest:
                if -n - a in rest and (-n-a != a or dic[a] > 1): 
                    res.add(tuple(sorted([n, a, -n-a])))
                    
        return [list(tri) for tri in res]

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums.sort()
        _dict = collections.defaultdict(int)
        for n in nums : _dict[n] += 1
        for key in _dict:
            if _dict[key] > 1:
                return True
        return False

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        nums.sort()
        for i, num in enumerate(nums):
            _num = nums[:i] + nums[i+1:]
            if num in _num:
                return True
        return False

class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        for i, num in enumerate(nums):
            for j, n in enumerate(nums):
                if (num == n) & (i != j) & (abs(i-j) <= k) :
                    return True
        return False

class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        _dict = collections.defaultdict(int)
        for n in nums : _dict[n] += 1
        for key in _dict:
            if _dict[key] >1 :
                _list = []
                for i in range(len(nums)):
                    if nums[i] == key:
                        _list.append(i)
                for m, _m in enumerate(_list):
                    for j, _j in enumerate(_list):
                        if (abs(_m - _j) <= k) & (m!=j):
                            return True
        return False


class Solution:
    def maxSubArray(self, nums):
        ans = []
        l = len(nums)
        if len(nums) > 1:
            for i in range(l):
                if len(nums[i:])>0:
                    ans.append(sum(nums[i:]))
                if len(nums[:i])>0:
                    ans.append(sum(nums[:i]))
                for j in range(l):
                    if len(nums[j:])>0:
                        ans.append(sum(nums[j:]))
                    if len(nums[:j])>0:
                        ans.append(sum(nums[:j]))
                    if (i!=j) & (len(nums[i:j]) > 0):
                        ans.append(sum(nums[i:j]))
            print(ans) ; exit()
            return max(ans)
        else:
            return nums[0] 

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums) <= 1: return nums[0]
        overall_max, current_max = nums[0], 0
        
        for i in range(len(nums)):
            current_max = max(nums[i], current_max + nums[i])
            overall_max = max(overall_max, current_max)
            
        return overall_max

class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        ans = []
        _list1 = nums[:n]
        _list2 = nums[n:]
        for x, y in zip(_list1, _list2):
            ans.append(x)
            ans.append(y)
        return ans
import collections
class Solution:
    def thirdMax(self, nums):
        if len(nums) <3:
            return max(nums)
        else:
            _num = sorted(nums)
            print(_num) ; exit()
            _dict = collections.defaultdict(int)
            for n in nums: _dict[n] += 1
            ans = [*_dict]
            print(ans) ; exit()
            if len(ans)>2:
                return ans[2]
            else:
                return max(ans)

Solution().thirdMax([1,2,2,5,3,5])

class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        answer = []
        _max = max(candies)
        _list = [x + extraCandies for x in candies]
        for can in _list:
            if can >= _max:
                answer.append(True)
            else:
                answer.append(False)
        return answer


class Solution:
    def defangIPaddr(self, address: str) -> str:
        return address.replace(".", "[.]")

class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        ans = 0
        _dict = collections.defaultdict(int)
        for n in nums : _dict[n] += 1
        for key in _dict:
            if _dict[key]>1:
                _list = []
                for i in range(len(nums)):
                    if nums[i] == key:
                        _list.append(i)
                for j in _list:
                    for k in _list:
                        if (j!=k) & (j<k):
                            ans += 1
        return ans

class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        num = [int(x) for x in str(n)]
        return self.prod(num) - sum(num)
        
    def prod(self, num):
        i = 1
        for k in num:
            i = i*k
        return i

class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        ans = []
        for i in range(len(nums)):
            a = 0
            for j in range(len(nums)):
                if (i!=j) & (nums[i] > nums[j]):
                    a += 1
            ans.append(a)
        return ans

class Solution:
    def restoreString(self, s: str, indices: List[int]) -> str:
        ans = []
        s_ = [str(x) for x in str(s)]
        for i in range(len(indices)):
            k = indices.index(i)
            ans.append(s_[k]) 
        return ''.join([str(elem) for elem in ans])

class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        _list = []
        _arr = sorted(arr)
        for i in range(1,len(_arr)):
            val = _arr[i-1] - _arr[i]
            _list.append(val)
        if min(_list) == max(_list):
            return True
        else:
            return False

class Solution:
    def reverseVowels(self, s: str) -> str:
        ans = [None]*len(s)
        _list = [str(x) for x in str(s)]
        vowels = ['a', 'e', 'i' , 'o', 'u', 'A', 'E', 'I' , 'O', 'U']
        vow = [x for x in _list if x in vowels]
        vow_flip = self.flip(vow)
        a = 0
        for j, letter in enumerate(_list):
            if letter not in vow:
                ans[j] = _list[j]
            else:
                ans[j] = vow_flip[a]
                a += 1
        return ''.join([str(x) for x in ans])
        
    def flip(self, vow):
        vow_flip = [None]*len(vow)
        for i in range(len(vow)):
            vow_flip[i] = vow[len(vow) - 1 - i]
        return vow_flip

class Solution:
    def reverseVowels(self, s: str) -> str:
        ans = [None]*len(s)
        _list = [str(x) for x in str(s)]
        vowels = ['a', 'e', 'i' , 'o', 'u', 'A', 'E', 'I' , 'O', 'U']
        vow = [x for x in _list if x in vowels]
        vow_flip = self.flip(vow)
        a = 0
        for j, letter in enumerate(_list):
            if letter not in vow:
                ans[j] = _list[j]
            else:
                ans[j] = vow_flip[a]
                a += 1
        return ''.join([str(x) for x in ans])
        
    def flip(self, vow):
        return [ele for ele in reversed(vow)]

class Solution:
    def reverseVowels(self, s: str) -> str:
        even_str,even = "", {'a','e','i','o','u','A','E','I','O','U'}
        for i in s:
            if i in even:
                even_str += i
        even_str = even_str[::-1]
        ans,idx = "",0
        for i in s:
            if i in even:
                ans += even_str[idx]
                idx += 1
            else:
                ans += i
        return ans
        
print(Solution().reverseVowels("A very NamrataGUPTA"))

class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        _len = len(s)
        k = _len//2
        i = 0
        while i < k:
            s[i], s[_len - 1 - i] = s[_len - 1 - i] ,  s[i]
            i += 1
            
        k = len(s)
        i = 0
        while i < k / 2:
            s[i], s[k-i-1] = s[k-i-1], s[i]
            i += 1
        
class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        if coordinates[1][0] != coordinates[0][0]:
            slope = (coordinates[1][1] - coordinates[0][1])/(coordinates[1][0] - 
                                                             coordinates[0][0])
            c = coordinates[1][1] - slope * coordinates[1][0]
            for points in coordinates:
                if points[1] != slope * points[0] + c :
                    return False
        else:
            for points in coordinates:
                c = coordinates[1][0]
                if points[0] != c:
                    return False
        return True

class Solution:
    def addToArrayForm(self, A: List[int], K: int) -> List[int]:
        s = ''.join([str(x) for x in A])
        s = int(s)
        s = s + K
        return [str(x) for x in str(s)]


class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        a = ''.join([str(x) for x in digits])
        a = int(a)
        a = a + 1
        return [str(x) for x in str(a)]

class Solution:
    def isMonotonic(self, A: List[int]) -> bool:
        x =1
        y =1
        for i in range(1,len(A)):
            if A[i-1] >= A[i]: 
                x += 1
        for i in range(1,len(A)):
            if A[i-1] <= A[i]:
                y += 1
        if (x == (len(A))) | (y == (len(A))) :
            return True
        else:
            return False

class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = [str(x) for x in str(s)]
        s = [x for x in s if x.isalnum()]
        s = [x.lower() for x in s]
        for i in range(len(s)):
            if s[i] != s[len(s) - 1 - i]:
                return False
        return True


class Solution:
    def validPalindrome(self, s: str) -> bool:
        a = 0
        s = [str(x) for x in str(s)]
        print(s)
        for i in range(len(s)):
            print('lll', i, 's', s)
            if s[i] != s[len(s) - 1 - i]:
                print(i)
                if (a <= 1) & (i < len(s) - 1):
                    if s[i+1] == s[len(s) - 1 - i]:
                        s.remove(s[i])
                        a += 1
                        break
                    if s[i] == s[len(s) - 2 - i]:
                        s.remove(s[len(s) - 1 - i])
                        a += 1
                        break
        print('sd', s)
        for i in range(len(s)):
            if s[i] != s[len(s) - 1 - i]:
                return False
        return True

print(Solution().validPalindrome("abc"))

class Solution(object):
    def validPalindrome(self, s):
        def is_pali_range(i, j):
            return all(s[k] == s[j-k+i] for k in range(i, j))

        for i in xrange(len(s) / 2):
            if s[i] != s[~i]:
                j = len(s) - 1 - i
                return is_pali_range(i+1, j) or is_pali_range(i, j-1)
        return True

class Solution:
    def validPalindrome(self, s: str) -> bool:
        x=s[::-1]
        a=''
        b=''
        for i in range(len(x)):
            if x[i]!=s[i]:
                a=x[0:i]+x[i+1:len(x)]
                b=s[0:i]+s[i+1:len(x)]
                if a!=a[::-1] and b!=b[::-1]:
                    return False
                else:
                    return True
        return True

class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        ans = ''
        _s = [s[i:i+k] for i in range(0, len(s), k)]
        for i, w in enumerate(_s):
            if i%2==0:
                ans += w[::-1]
            else:
                ans += w
        return ans

class Solution:
    def reverseWords(self, s: str) -> str:
        ans = []
        _s = s.split(' ')
        for i in _s:
            ans.append(i[::-1])
        return ' '.join(ans)
        

class Solution:
    def isPrefixOfWord(self, sentence: str, searchWord: str) -> int:
        sent = sentence.split()
        for i , sen in enumerate(sent):
            x = [str(x) for x in str(sen)]
            ans = ''
            for j in x:
                ans += j
                if ans == searchWord:
                    return i+1
        return -1

class Solution:
    def complexNumberMultiply(self, a: str, b: str) -> str:
        ans = []
        a = a.replace('i','')
        b = b.replace('i','')
        x = a.split("+")
        y = b.split("+")
        for i in range(len(x)):
            for j in range(len(y)):
                ans.append(int(x[i])*int(y[j]))
        m = ans[0] - ans[3]
        n = ans[1] + ans[2]
        return str(m)+ "+" + str(n) + "i"


class Solution:
    def numUniqueEmails(self, emails):
        ans = []
        for email in emails:
            if '+' in email:
                k = email.split('+')
                l = k[-1].split('@')
                email = k[0].replace('.', '') + '@' + l[-1]
            else:
                k = [email]
                print('k', k)
                l = k[0].split('@')
                email = l[0].replace('.', '') + '@' + l[-1]
            ans.append(email)
        print(ans)
        output = []
        for a in ans:
            if a not in output:
                output.append(a)
        return len(output)

print(Solution().numUniqueEmails(["test.email+alex@leetcode.com", "test.email@leetcode.com"]))

class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        para = paragraph.lower()
        para = para.split(' ')
        case = ['!', '?', ',',';','.',"'"]
        par = []
        for p in para:
            for c in case:
                p = p.replace(c,'')
            par.append(p)
        print(par)
        _dict = collections.defaultdict(int)
        for n in par: _dict[n] += 1
        _dic = {k:v for k, v in sorted(_dict.items(), key = lambda item: item[1], 
                                      reverse = True)}
        for key in _dic:
            if key not in banned:
                return key

class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        d=collections.defaultdict(int)
        print('d', d)
        paragraph = re.sub(r'[^a-zA-Z ]',' ',paragraph)
        print('paragraph', paragraph)
        for word in re.split(r' {1,}',paragraph.lower()):
            if word not in banned:
                d[word]+=1
        print('d', d)
        return max(d,key=d.get)

class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if log(n,2)%1 < 1e-5:
            return True
        else:
            return False
        print(log(536870912,2))

class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        power = 1
        while power < n:
            power *= 2
        if power == n:
            return True
        return False

class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        s = [str(x) for x in str(s)]
        t = [str(x) for x in str(t)]
        s_dict = collections.defaultdict(int)
        t_dict = collections.defaultdict(int)
        for n in s: s_dict[n] += 1
        for n in t: t_dict[n] += 1
        for key in t_dict:
            if key in s_dict:
                if t_dict[key] != s_dict[key]:
                    return key
            else:
                return key

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        _dict = collections.defaultdict(int)
        for n in nums: _dict[n] += 1
        for key in _dict:
            if _dict[key] == 1:
                return key

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        nums = sorted(nums)
        if nums[0] != 0:
            return 0
        elif len(nums) > 1:
            for i in range(1,len(nums)):
                if nums[i -1] + 1 != nums[i]:
                    return nums[i-1] + 1
        return nums[-1] + 1

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        _dict = collections.defaultdict(int)
        for n in nums: _dict[n] += 1
        for key in _dict:
            if _dict[key] > 1:
                return key

class Solution:
    def findComplement(self, num: int) -> int:
        binary = ''
        if num !=0:
            while num > 0:
                if (num % 2 == 0):
                    binary += '0'
                    num = num//2
                else:
                    binary += '1'
                    num = num//2
            binary = binary[::-1]
            _list = [str(x) for x in str(binary)]
            ans = []
            _ans = 0
            for i in range(len(_list)):
                if _list[i] == '1':
                    ans.append(0)
                else:
                    ans.append(1)
            for i in range(len(ans)):
                _ans = _ans + 2**(len(ans) - 1 - i) * int(ans[i])
            return _ans
        return 0

class Solution:
    def bitwiseComplement(self, N: int) -> int:
        num = N
        binary = ''
        if num !=0:
            while num > 0:
                if (num % 2 == 0):
                    binary += '0'
                    num = num//2
                else:
                    binary += '1'
                    num = num//2
            binary = binary[::-1]
            _list = [str(x) for x in str(binary)]
            ans = []
            _ans = 0
            for i in range(len(_list)):
                if _list[i] == '1':
                    ans.append(0)
                else:
                    ans.append(1)
            for i in range(len(ans)):
                _ans = _ans + 2**(len(ans) - 1 - i) * int(ans[i])
            return _ans
        return 1

class Solution:
    def hammingWeight(self, n: int) -> int:
        n = bin(n)
        print(n)
        _list = [str(x) for x in str(n)]
        print(_list)
        _dict = collections.defaultdict(int)
        for n in _list: _dict[n] += 1
        print(_dict)
        for key in _dict:
            if key == '1':
                return _dict[key]
        return 0

class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        _n = self._binary(n)
        _list = [str(x) for x in _n]
        for i in range(1,len(_list)):
            if _list[i-1] == _list[i]:
                return False
        return True
    
    
    def _binary(self, n):
        ans = ''
        if n != 0:
            while n >0 :
                if n%2 == 0:
                    ans += '0'
                    n = n//2
                else:
                    ans += '1'
                    n = n//2
            return ans[::-1]
        return 0

class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        x = '{:032b}'.format(x)
        y = '{:032b}'.format(y)
        x = [str(i) for i in str(x)]
        y = [str(i) for i in str(y)]
        a = 0
        for i,j in zip(x,y):
            if i != j:
                a += 1
        return a

class Solution:
    def totalHammingDistance(self, nums: List[int]) -> int:
        _list = []
        a = 0
        for i in range(1,len(nums)):
            for j in range(i,len(nums)):
                b = self.hamdist([str(x) for x in str('{:032b}'.format(nums[i-1]))]
                                ,[str(x) for x in str('{:032b}'.format(nums[j]))])
                a += b
        return a
                
    def hamdist(self, m,n):
        k = 0
        for i, j in zip(m,n):
            if i != j:
                k += 1
        return k
                
class Solution:
    def reorderSpaces(self, text: str) -> str:
        _list = text.split()
        b = text.count(' ')
        a = len(_list)
        if a > 1:
            m = b//(a - 1)
            n = b%(a - 1)
        else:
            m,n = 0, b
        space = ' '
        ans = ''
        for i in range(len(_list)):
            ans += _list[i]
            if i < len(_list) - 1:
                ans += m*space
        return ans + n*space

class Solution:
    def mySqrt(self, x: int) -> int:
        return int(x**(1/2))

class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if (int(num**(1/2)) - num**(1/2) == 0):
            return True
        return False


class Solution:
    def kSmallestPairs(self, nums1, nums2, k):
        if (nums1 == []) | (nums2 == []):
            return []
        _pair = [None]*k
        _sum = [None]*k
        a = 0
        for i in range(len(nums1)):
            for j in range(len(nums2)):
                _plist = [nums1[i], nums2[j]]
                _s = nums1[i] + nums2[j]
                if a == 0:
                    _pair[0] = _plist
                    _sum[0] = _s
                    print('werttt', a, _pair, _sum)
                else:
                    for m in reversed(range(a)):
                        print('a', a,'m', m)
                        if _s < _sum[m]:
                            print('_s', _s, '_sum[m]', _sum[m])
                            _sum[m+1] = _sum[m]
                            _pair[m+1] = _pair[m]
                            _sum[m] = _s
                            _pair[m] = _plist
                            if a < k:
                                a += 1
                            print('weee', a, _pair, _sum)
                        else:
                            _pair[a] = _plist
                            _sum[a] = _s
                            print('teyeueu', a, _pair, _sum)
                if a < k:
                    a += 1
        return _pair
                    
class Solution:
    def countPrimes(self, n: int) -> int:
        if (n == 0) | (n == 1):
            return 0
        elif n == 2:
            return 0
        else:
            a = 0
            for i in range( 3,n):
                m = self.is_prime(i)
                a += m
        return a + 1
    
    def is_prime(self, n):
        sqr = int((n)**(1/2)) + 1
        for divisor in range(2, sqr):
            if n % divisor == 0:
                return 0
        return 1
            
class Solution:
    def isBoomerang(self, points: List[List[int]]) -> bool:
        p1, p2, p3 = points
        if ((p1[0] == p2[0]) & (p1[1] == p2[1])) | ((p1[0] == p3[0]) & (p1[1] == p3[1])) | ((p2[0] == p3[0]) & (p2[1] == p3[1])):
            return False
        if (points[1][0] - points[0][0] != 0):
            print('ger')
            slope = (points[1][1] - points[0][1])/(points[1][0] - points[0][0])
            c = points[1][1] - slope*points[1][0]
            if (points[2][1] == (slope*points[2][0] + c)):
                return False
        else:
            print('gerer')
            if (points[2][0] == points[1][0]):
                return False
        return True


            
# Solution().kSmallestPairs([1,1,2], [1,2,3], 10)

class Solution:
    def kSmallestPairs(self, nums1, nums2, k):
        l=[]
        nums1.sort()
        nums2.sort()
        for i in range(len(nums1)):
            for j in range(len(nums2)):
                l.append([nums1[i],nums2[j]])
        
        l.sort(key = lambda x: (x[0] + x[1]))
        final=[]
        print(l)
        for i in range(k):
            if i < len(l):
                final.append(l[i])
        return final

Solution().kSmallestPairs([1,1,2], [1,2,3], 10)

class Solution:
    def checkIfExist(self, arr: List[int]) -> bool:
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                if (i!=j) & ((arr[i] == 2*arr[j]) | (2*arr[i] == arr[j])) :
                    return True
        return False

#The isBadVersion API is already defined for you.
#@param version, an integer
#@return an integer
#def isBadVersion(version):

class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        i, j = 0, n
        while i < j:
            mid = (j+i)//2
            if isBadVersion(mid):
                j = mid
            else:
                i = mid + 1
        return i

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if len(nums) == 1:
            if nums[0] >= target:
                return 0
            else:
                return 1
        for i in range(len(nums)):
            if nums[i] >= target:
                return i
        return len(nums)

class Solution:
    def trimMean(self, arr: List[int]) -> float:
        arr = sorted(arr)
        i,j = int(.05*len(arr)), int(.95*len(arr))
        ans = arr[i:j]
        return sum(ans)/len(ans)

class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """
        for i in range(1,len(arr)):
            if arr[i-1] == 0:
                for j in reversed(range(i+1,len(arr))):
                    arr[j] = arr[j-1]
                arr[i] = -99
        for i in range(len(arr)):
            if arr[i] == -99:
                arr[i] = 0
        return arr

class Solution:
    def frequencySort(self, nums: List[int]) -> List[int]:
        _d = collections.defaultdict(int)
        for n in nums: _d[n] += 1
        _sds = {k: _d[k] for k in sorted(_d, reverse = True)}
        _sd = {k:v for k,v in sorted(_sds.items(), key = lambda item: item[1])}
        ans = []
        for key in _sd:
            ans += [key]*_sd[key]
        return ans

class Solution:
    def matrixReshape(self, nums: List[List[int]], r: int, c: int) -> List[List[int]]:
        _list = []
        ans = []
        for n in nums:
            _list += n
        if len(_list) != r*c:
            return nums
        else:
            for i in range(0,r):
                a = _list[i*c:i*c+c]
                ans.append(a)
        return ans

class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        a = 0
        for i in range(len(arr)):
            for j in range(len(arr)+1):
                print(i,j)
                if len(arr[i:j])%2!=0:
                    print(arr[i:j])
                    a += sum(arr[i:j])
        return a

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices)<=1:
            return 0
        ans = [prices[1]-prices[0]]
        for i in range(1,len(prices)):
            for j in range(i,len(prices)):
                profit = prices[j] - prices[i-1]
                if profit > ans[0]:
                    ans[0] = profit
        if ans[0]<0:
            return 0
        else:
            return ans[0]

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        result = 0
        sell = 0
        for i in reversed(range(len(prices))):
            sell = max(sell, prices[i])
            result = max(result, sell - prices[i])
        return result

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        lowestPrice=float('inf')
        maxProfit=0
        
        for currentPrice in prices:
            if currentPrice<lowestPrice:
                lowestPrice=currentPrice
                
            if currentPrice>lowestPrice:
                localMaxProfit=currentPrice-lowestPrice
                
                if localMaxProfit > maxProfit:
                    maxProfit=localMaxProfit
                    
        return maxProfit

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) <= 1:
            return 0
        minimum = prices[0]
        res = 0
        for i in range(1, len(prices)):
            res = max(res, prices[i] - minimum)
            minimum = min(minimum, prices[i])
        
        return res

class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        first = cost[0]
        second = cost[1]
        for i in range(2,len(cost)):
            curr = cost[i] + min(first, second)
            first = second
            second = curr
        return min(first, second)

def solve(n):
   
  #Base case
  if n < 1:
    return 0
  if n == 1:
    return 1
   
  return (solve(n - 1) +
          solve(n - 3) +
          solve(n - 5))

print(solve(13))

class Solution:
    def decompressRLElist(self, nums: List[int]) -> List[int]:
        ans = []
        for i in range(0,len(nums)-1,2):
            a = [nums[i+1]]*nums[i]
            ans += a
        return ans

class Solution:
    def createTargetArray(self, nums: List[int], index: List[int]) -> List[int]:
        target = []
        for x, y in zip(nums, index):
            target.insert(y,x)
        return target

class Solution:
    def flipAndInvertImage(self, A: List[List[int]]) -> List[List[int]]:
        ans = []
        for _list in A:
            _list = self.flip(_list)
            for i in range(len(_list)):
                if _list[i] == 0:
                    _list[i] = 1
                else:
                    _list[i] = 0
            ans.append(_list)
        return ans
                    
    def flip(self, vow):
        return [ele for ele in reversed(vow)]    

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        x = len(nums)
        _dict = collections.defaultdict(int)
        for n in nums: _dict[n] += 1
        for key in _dict:
            if _dict[key] > x/2:
                return key

class Solution:
    def countOdds(self, low: int, high: int) -> int:
        if (low%2 == 0) & (high%2 == 0):
            return int((high - low)/2)
        elif (low%2 != 0) & (high%2 != 0):
            return int(((high - low)/2) + 1)
        else:
            return int(((high - low)//2) + 1)

class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        a = 0
        _j = [str(x) for x in str(J)]
        _S = [str(x) for x in str(S)]
        for x in _S:
            if x in _j:
                a += 1
        return a

class Solution:
    def numberOfSteps (self, num: int) -> int:
        a = 0
        while num > 0:
            if num%2 == 0:
                num = int(num/2)
                a += 1
            else:
                num = num - 1
                a += 1
        return a 

class Solution:
    def numSubarraysWithSum(self, A: List[int], S: int) -> int:
        a = 0
        for i in range(len(A)):
            for j in range(i,len(A)):
                _l = A[i:j+1]
                # print(_l)
                if sum(_l) == S:
                    a += 1
        return a

class Solution:
    def numSubarraysWithSum(self, A: List[int], S: int) -> int:
        i = 0
        currsum = 0
        cnt = 0
        wt = 1
        for j in range(len(A)):
            currsum += A[j]
            if currsum == S:
                cnt += wt
                while i < j and A[i] == 0:
                    cnt += 1
                    i += 1
                    wt +=1
            elif currsum > S:
                wt = 1
                while i < j and currsum > S:
                    currsum -= A[i]
                    i += 1
                if currsum == S:
                    cnt += 1
                    while i<j and A[i] == 0:
                        cnt += 1
                        i += 1
                        wt +=1
        return cnt

class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        for i in range(len(nums)):
            if nums[i]%2 != 0:
                nums[i] = 1
            else:
                nums[i] = 0
        i = 0
        currsum = 0
        wt = 1
        cnt = 0
        for j in range(len(nums)):
            currsum += nums[j]
            if currsum == k:
                cnt += wt
                while i < j and nums[i] == 0:
                    cnt += 1
                    i += 1
                    wt += 1
            elif currsum > k:
                wt = 1
                while i < j and currsum > k:
                    currsum -= nums[i]
                    i += 1
                if currsum == k:
                    cnt += 1
                    while i < j and nums[i] == 0:
                        cnt += 1
                        i += 1
                        wt += 1
        return cnt
               
class Solution:
    def sortedSquares(self, A: List[int]) -> List[int]:
        for i in range(len(A)):
            A[i] = A[i]**2
        return sorted(A)

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        j = 0
        for i in range(m,len(nums1)):
            nums1[i] = nums2[j]
            j += 1
        return nums1.sort()

class Solution:
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        even = []
        odd = []
        for i in range(len(A)):
            if A[i]%2 == 0:
                even.append(A[i])
            else:
                odd.append(A[i])
        return (even + odd)

class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        for x in nums:
            if x == target:
                return True
        return False

class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        s = [str(x) for x in str(s)]
        t = [str(x) for x in str(t)]
        a = -1
        c = 0
        for x in s:
            if x not in t:
                return False
            else:
                m = [i for i, val in enumerate(t) if val == x]
                for j in m:
                    if j > a:
                        a = j
                        c += 1
                        break
        if c != len(s):
            return False
        else:
            return True

class Solution:
    def numMatchingSubseq(self, S: str, words: List[str]) -> int:
        S = [str(x) for x in str(S)]
        cnt = 0
        for t in words:
            n = 0
            a = -1
            t = [str(x) for x in str(t)]
            for x in t:
                if x not in S:
                    break
                else:
                    m = [i for i, val in enumerate(S) if val == x]
                    for j in m:
                        if j > a:
                            a = j
                            n += 1
                            break
            if n == len(t):
                cnt += 1
        return cnt

class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i = 0
        j = 0
        while (i < len(s) and j < len(t)):
            if s[i] == t[j]:
                i += 1
            j += 1
        if i == len(s):
            return True
        else:
            return False

class Solution:
    def numMatchingSubseq(self, S: str, words: List[str]) -> int:
        _dict = collections.Counter(words)
        cnt = sum([_dict[word] for word in _dict if self.subseq(word,S)])
        return cnt
    
    def subseq(self, t, S):
        i = 0
        j = 0
        while (i < len(t) and j < len(S)):
            if t[i] == S[j]:
                i += 1
            j += 1
        if i == len(t):
            return True
        else:
            return False

class Solution:
    def matrixBlockSum(self, mat: List[List[int]], K: int) -> List[List[int]]:
        ans = [[0 for i in range(len(mat[0]))] for _ in range(len(mat))]
        for i in range(len(ans)):
            for j in range(len(ans[0])):
                val = 0
                for r in range(i-K,i+K+1):
                    for c in range(j-K,j+K+1):
                        if r>=0 and r< len(mat) and c>= 0 and c< len(mat[0]):
                            val += mat[r][c]
                ans[i][j] = val
        return ans

class Solution:
    def matrixBlockSum(self, mat: List[List[int]], K: int) -> List[List[int]]:
        ans = [[0 for i in range(len(mat[0]))] for _ in range(len(mat))]
        for i in range(len(ans)):
            for j in range(len(ans[0])):
                val = 0
                r1 = max(i-K,0)
                r2 = min(i+K+1, len(mat))
                c1 = max(j-K,0)
                c2 = min(j+K+1, len(mat[0]))
                for r in range(r1,r2):
                    for c in range(c1,c2):
                        val += mat[r][c]
                ans[i][j] = val
        return ans

class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i >0 and j>0:
                    grid[i][j] = min(grid[i][j] + grid[i][j-1], 
                                     grid[i][j] + grid[i-1][j])
                elif i > 0:
                    grid[i][j] += grid[i-1][j]
                elif j > 0:
                    grid[i][j] += grid[i][j-1]
        return grid[-1][-1]

class Solution:
    def minFallingPathSum(self, A: List[List[int]]) -> int:
        for i in range(1,len(A)):
            for j in range(len(A[0])):
                if j>0 and j < (len(A[0])-1):
                    A[i][j] = min(A[i][j] + A[i-1][j], A[i][j]+ A[i-1][j-1], 
                                 A[i][j]+ A[i-1][j+1])
                elif j==0:
                    A[i][j] = min(A[i][j] + A[i-1][j], A[i][j]+ A[i-1][j+1])
                elif j == len(A[0]) - 1:
                    A[i][j] = min(A[i][j] + A[i-1][j], A[i][j]+ A[i-1][j-1])
        return min(A[-1])

class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        for i in range(len(nums)):
            j = 0
            currsum = 0
            currsum = sum(nums[:i+1])
            if k != 0:
                if currsum%k == 0 and i > 0:
                    return True
                else:
                    while j < i-1:
                        currsum -= nums[j]
                        j += 1
                        if currsum%k == 0:
                            return True
            else:
                if currsum == 0 and i > 0:
                    return True
                else:
                    while j < i-1:
                        currsum -= nums[j]
                        j += 1
                        if currsum == 0:
                            return True             
        return False
                
            
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        nums = sorted(nums)
        val = 0
        for i in range(1,len(nums),2):
            currval = min(nums[i-1], nums[i])
            val += currval
        return val

class Solution:
    def getMaximumGenerated(self, n: int) -> int:
        ans = [None]*(n+1)
        if n > 1:
            ans[0] = 0
            ans[1] = 1
            for i in range(2,n+1):
                if i%2 == 0:
                    ans[i] = ans[i//2]
                else:
                    ans[i] = ans[i//2] + ans[(i//2) + 1]
            return max(ans)
        elif n == 1:
            return 1
        else:
            return 0

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        curval = 0
        for i in range(len(nums)):
            j = 0
            while j < i:
                if (nums[i]-1)*(nums[j]-1) > curval:
                    curval = (nums[i]-1)*(nums[j]-1)
                j += 1
        return curval

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        nums = sorted(nums)
        ans = (nums[-1] - 1)*(nums[-2] - 1)
        return ans

class Solution:
    def oddCells(self, n: int, m: int, indices: List[List[int]]) -> int:
        _l = [[0 for x in range(m)] for _ in range(n)]
        for index in indices:
            i , j = index
            k = 0
            while k < m:
                _l[i][k] += 1
                k += 1
            l = 0
            while l < n:
                _l[l][j] += 1
                l += 1
        ans = 0
        for i in range(n):
            for j in range(m):
                if _l[i][j]%2 == 1:
                    ans += 1
        return ans

class Solution:
    def minFlipsMonoIncr(self, S: str) -> int:
        p = [0]
        for x in S:
            p.append(p[-1] + int(x))
        for i in range(len(p)):
            p[i] = p[i] + (len(S) - i) - (p[-1] - p[i])
        return min(p)

class Solution:
    def dominantIndex(self, nums: List[int]) -> int:
        x = max(nums)
        x_index = nums.index(x)
        for i in range(len(nums)):
            if 2 * nums[i] > x and i != x_index:
                return -1
        return x_index

class Solution:
    def numSmallerByFrequency(self, queries: List[str], words: List[str]) -> List[int]:
        q = [None]*len(queries)
        for i in range(len(queries)):
            queries[i] = sorted(queries[i])
            _d = collections.Counter(queries[i])
            q[i] = _d[list(_d.keys())[0]]
        w = [None]*len(words)
        for i in range(len(words)):
            words[i] = sorted(words[i])
            _d = collections.Counter(words[i])
            w[i] = _d[list(_d.keys())[0]]
        ans = [None]*len(q)
        for x in range(len(q)):
            a = 0
            for y in range(len(w)):
                if q[x] < w[y]:
                    a += 1
            ans[x] = a
        return ans

class Solution:
    def kLengthApart(self, nums: List[int], k: int) -> bool:
        # p = [0]
        # for x in nums:
        #     p.append(p[-1] + int(x))
        p = [i for i,x in enumerate(nums) if x ==1]
        print(p)
        for i in range(1,len(p)):
            m = p[i] - p[i-1] - 1
            if m < k :
                return False
        return True

a = [[1,2,3],[0,1,3],[2,4,1]]
k = []
for x in a:
    print(x)
    if x[0] < 2:
        k.append(x)
        print(k)
a = [e for e in a if e not in k]
print(a)

class Solution:
    def filterRestaurants(self, restaurants: List[List[int]], veganFriendly: int, maxPrice: int, maxDistance: int) -> List[int]:
        a = []
        if veganFriendly == 1:
            for x in restaurants:
                if x[2] == 0:
                    a.append(x)
        restaurants = [e for e in restaurants if e not in a]
        a = []
        for x in restaurants:
            print(x[3]) 
            if x[3] >  maxPrice:
                a.append(x)
        restaurants = [e for e in restaurants if e not in a]
        a = []
        for x in restaurants:
            if x[4] > maxDistance:
                a.append(x)
        restaurants = [e for e in restaurants if e not in a]
        restaurants = sorted(restaurants, key = lambda restaurants: restaurants[0],
                             reverse = True)
        restaurants = sorted(restaurants, key = lambda restaurants: restaurants[1],
                             reverse = True)
        ans = []
        for x in restaurants:
            ans.append(x[0])
        return ans
        
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        cnt = 0
        for j in range(len(time)):
            i = 0
            while i < j:
                if (time[i] + time[j])%60 == 0:
                    cnt += 1
                i += 1
        return cnt

class Solution:
    def partitionDisjoint(self, A: List[int]) -> int:
        left_max = A[0]
        next_max = 0
        res = 0
        for i in range(1, len(A)):
            if A[i] > next_max:
                next_max = A[i]
            if A[i] < left_max:
                res = i
                if next_max > left_max:
                    left_max = next_max
        return res + 1

class RecentCounter:

    def __init__(self):
        self.slider = []

    def ping(self, t: int) -> int:
        self.slider.append(t)
        k = []
        for i in range(len(self.slider)):
            if self.slider[i] < t-3000:
                k.append(self.slider[i])
        self.slider = [x for x in self.slider if x not in k]
        return len(self.slider)

class RecentCounter:

    def __init__(self):
        self.slider = []

    def ping(self, t: int) -> int:
        self.slider.append(t)
        while self.slider[0] < t-3000:
            self.slider.pop(0)
        return len(self.slider)

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i = 0
        while i < len(nums):
            if nums[i] == val:
                nums.pop(i)
                i = 0
            else:
                i += 1

class Solution:
    def replaceElements(self, arr: List[int]) -> List[int]:
        ans = [None]*len(arr)
        for i in range(len(arr)-1):
            ans[i] = max(arr[i+1:])
        ans[-1] = -1
        return ans

class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        i = 0
        p = [0]
        while i < len(nums):
            p.append(p[-1] + int(nums[i]))
            i += 1
        x = 0
        cur = 0
        best = 0
        for j in range(1,len(p)):
            if p[j] == p[j-1] + 1:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return best
                
class Solution:
    def maxPower(self, s: str) -> int:
        s = ' ' + s
        cnt = 0
        best = 0
        for i in range(1,len(s)):
            if s[i-1] == s[i]:
                cnt += 1
                if cnt > best:
                    best = cnt
            else:
                cnt = 0
        return best + 1

class Solution:
    def finalPrices(self, prices: List[int]) -> List[int]:
        ans = [None]*len(prices)
        for i in range(len(prices)):
            j = i + 1
            while j < len(prices):
                if prices[j] <= prices[i]:
                    ans[i] = prices[i] - prices[j]
                    break
                else:
                    j += 1
            if ans[i] == None:
                ans[i] = prices[i]
        ans[-1] = prices[-1]
        return ans

class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        missing = set(range(1,len(nums)+1))
        nums = set(nums)
        missing = list(missing.difference(nums))
        return missing

class Solution:
    def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
        a = collections.Counter(arr)
        t = collections.Counter(target)
        for key in a:
            if key in t:
                if a[key] != t[key]:
                    return False
            else:
                return False
        return True

class Solution:
    def sortArrayByParityII(self, A: List[int]) -> List[int]:
        ans = []
        odd = [x for x in A if x%2 != 0]
        even = [x for x in A if x%2 == 0]
        for i,j in zip(odd, even):
            ans.append(j)
            ans.append(i)
        return ans
        
class Solution:
    def findLucky(self, arr: List[int]) -> int:
        arr = sorted(arr, reverse = True)
        a = collections.Counter(arr)
        for key in a:
            if key == a[key]:
                return key
        return -1

class Solution:
    def canFormArray(self, arr: List[int], pieces: List[List[int]]) -> bool:
        ans = []
        for x in pieces:
            if x in [arr[i:len(x)+i] for i in range(len(arr))]:
                ans += x
            else:
                return False
        if len(ans) != len(arr):
            return False
        return True

class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        a = 0
        for x in nums:
            if len(str(x))%2 == 0:
                a += 1
        return a

class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        ans = []
        a = collections.Counter(nums)
        for key in a:
            if a[key] == 2:
                ans.append(key)
        return ans

class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        m = []
        a1 = collections.Counter(arr1)
        for x in arr2:
            if x in arr1:
                k = [x]*a1[x]
                m += k
        n = [x for x in arr1 if x not in arr2]
        n = sorted(n)
        return m+n


class Solution:
    def sumZero(self, n: int) -> List[int]:
        result = []
        if n % 2 == 1 :
            result.append(0)
        
        for i in range(1, n//2 +1):
            result.append(i)
            result.append(-i)
        return result 

class Solution:
    def findKthPositive(self, arr: List[int], k: int) -> int:
        a = []
        for i in range(arr[-1] + k):
            if i+1 not in arr and len(a) < k:
                a.append(i+1)
        return a[-1]

class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        a = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] < 0:
                    a += len(grid[0]) - j
                    break
        return a

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        a = [x for x in nums if x >0]
        if len(a) > 0:
            for i in range(max(a) + 1):
                if i + 1 not in a:
                    return i+1
        else:
            return 1

class Solution:
    def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
        a = 0
        for i, j in zip(startTime, endTime):
            if queryTime >= i and queryTime <= j:
                a += 1
        return a     
class Solution:
    def findSpecialInteger(self, arr: List[int]) -> int:
        l = len(arr)
        d = collections.Counter(arr)
        for key in d:
            if d[key]/l > 0.25:
                return key

