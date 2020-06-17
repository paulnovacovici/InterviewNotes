# LIS (Longest Increasing/Descreasing Subsequence)
**How to approach:** Start small and expand outwards keeping track of longest subsequence at i.

**Run time:** O(N^2)

## Examples:

**Description:** Given an unsorted array of integers, find the length of longest increasing subsequence.
Given an unsorted array of integers, find the length of longest increasing subsequence.
```
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        N = len(nums)
        
        # Recursive
        def dp(i, j):
            if i >= N or j > i:
                return 0
            
            if i == j:
                return max(1 + dp(i+1, j), dp(i+1, j+1))
            
            if nums[i] > nums[j]:
                return max(1 + dp(i+1, i), dp(i+1, j))
            
            return dp(i+1, j)
        
        return dp(0,0)
 -------------------------------------------------------------- 
        # DP          
        dp = [[0] * (N+1) for _ in range(N+1)]
    
        for i in range(N - 1, -1, -1):
            dp[i][i] = max(1 + dp[i+1][i], dp[i+1][i+1])
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i][j] = max(1 + dp[i+1][i], dp[i+1][j])
                else:
                    dp[i][j] = dp[i+1][j]
                    
        return dp[0][0]    
---------------------------------------------------------------  
        # Optimized DP
        if N == 0:
            return 0
            
        dp = [1] * N
        ret = 1
        
        for i in range(1, N):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[j] + 1, dp[i])
                    
            ret = max(ret, dp[i])
            
        return ret
```

**Description:** Given a set of distinct positive integers, find the largest subset such that every pair (Si, Sj) of elements in this subset satisfies:

Si % Sj = 0 or Sj % Si = 0.

If there are multiple solutions, return any subset is fine. [Leetcode](https://leetcode.com/problems/largest-divisible-subset/)

```
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        ret = 0
        N = len(nums)
        nums = sorted(nums)
        
        if N == 0:
            return []
        
        pointer = [i for i in range(N)]
        dp = [1 for _ in range(N)]
        maxvalue, maxindex = 0, 0
        for i in range(1, N):
            m = 1
            p = i
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    if dp[j] + 1 > m:
                        m = dp[j] + 1
                        p = j
                        
            dp[i] = m
            if m > maxvalue:
                maxvalue = m
                maxindex = i
            pointer[i] = p
            
        ans = []
        while maxindex != pointer[maxindex]:
            ans.append(nums[maxindex])
            maxindex = pointer[maxindex]
        
        ans.append(nums[maxindex])
        return ans[::-1]
```

# Sliding Items
**How to approach these problems:** If there's ever a question where I need to slide elements to the end of the array, then keep a total of the correct elements and always switch the correct element to the total, and skip bad elements.

## Examples: 
```
def removeDuplicates(self, nums: List[int]) -> int:
    if len(nums) == 0:
        return 0

    i = 0
    total = 0
    count = 0
    cur = nums[0]

    while i < len(nums):
        if nums[i] != cur:
            cur = nums[i]
            count = 1
            nums[total] = nums[i]
            total += 1
        elif count < 2:
            nums[total] = nums[i]
            total += 1
            count += 1
        i += 1
```
# Dynamic Programming
**Notes on how to approach dp problems:** If you can simplify the recursion to only using indexes in the recursive step then you can make it faster by using dynamic programming. Look at the dependencies and build from leaf of recursion tree

- 1D example:
```
# Solving with recursion
def numDecodings(self, s: str) -> int:   
        def _numDecodings(i):
            if i == len(s): return 1
            if s[i] == '0': return 0
            if i == len(s) - 1: return 1
            num = int(s[i:i+2])
            if 9 < num <= 26:
                return _numDecodings(i+1) + _numDecodings(i+2)
            else:
                return _numDecodings(i+1)
            
        return _numDecodings(0)

# Solving using dp after simplifying recursion
def numDecodings(self, s: str) -> int:  
        dp = [0] * (len(s) + 1)
        
        dp[len(s)] = 1
        
        i = len(s) - 1
        while i >= 0:
            if s[i] == '0':
                dp[i] = 0
            elif i == len(s) - 1:
                dp[i] = 1
            elif int(s[i:i+2]) > 26:
                dp[i] = dp[i+1]
            else:
                dp[i] = dp[i+1] + dp[i+2]
            i -= 1
        return dp[0]
```

- Re-using space example:
```
"""
Desc: Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.

For example, given the following triangle:
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
"""
class Solution:
    # Brute force
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        def _minimumTotal(lvl, i):
            if lvl >= len(triangle):
                return 0
            
            return triangle[lvl][i] + min(_minimumTotal(lvl+1, i), _minimumTotal(lvl+1, i+1))
        
        return _minimumTotal(0, 0)
    
    # Memoization
    def minimumTotal(self, triangle: List[List[int]]) -> int:        
        for lvl in range(len(triangle)-2, -1, -1):
            for i in range(len(triangle[lvl]) - 1, -1, -1):
                triangle[lvl][i] = triangle[lvl][i] + min(triangle[lvl+1][i], triangle[lvl+1][i+1])
    
        return triangle[0][0]
```

**Description (Hard):**  Given an array of integers cost and an integer target. Return the maximum integer you can paint under the following rules:

The cost of painting a digit (i+1) is given by cost[i] (0 indexed).
The total cost used must be equal to target.
Integer does not have digits 0.
Since the answer may be too large, return it as string.

If there is no way to paint any integer given the condition, return "0".

```
class Solution:
    def largestNumber(self, cost: List[int], target: int) -> str:
        def dp(target):
            if target < 0:
                return -1
            
            if target == 0:
                return 0
            
            return max(dp(target - c) * 10 + num + 1 for num, c in enumerate(cost))
        
        dp = [0] + [-1] * target
        
        for t in range(target+1):
            for num, c in enumerate(cost):
                if t - c >= 0:
                    dp[t] = max(dp[t - c] * 10 + num + 1, dp[t])
        return str(max(dp[target], 0))
```

# Minimax (Red Black Trees):
**Description:** Given an array of scores that are non-negative integers. Player 1 picks one of the numbers from either end of the array followed by the player 2 and then player 1 and so on. Each time a player picks a number, that number will not be available for the next player. This continues until all the scores have been chosen. The player with the maximum score wins.

Given an array of scores, predict whether player 1 is the winner. You can assume each player plays to maximize his score.
[Leetcode](https://leetcode.com/problems/predict-the-winner/)

**Answer progression from naive solution brute force to DP:**
```
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        N = len(nums)

Naive:        
----------------------------------------------------------------------------------      
        def helper(i, j, player1Turn):
            if i > j:
                return 0
            
            if player1Turn:
                return max(nums[i] + helper(i+1, j, False), nums[j] + helper(i, j-1, False))
            else:
                return min(helper(i+1, j, True) - nums[i],  helper(i, j-1,  True) - nums[j])
Memoization:
----------------------------------------------------------------------------------           
        @functools.lru_cache(None)
        def helper(i, j):
            if i > j:
                return 0
            
            return max(nums[i] - helper(i+1, j), nums[j] - helper(i, j-1))
DP:
----------------------------------------------------------------------------------          
        dp = [[0] * (N+1) for _ in range(N+1)]
        
        for i in range(N-1, -1, -1): 
            for j in range(i, N):
                dp[i][j] = max(nums[i] - dp[i+1][j], nums[j] - dp[i][j-1])
                
        return dp[0][N-1] >= 0
```

# Topological Sort:
**Description:** There is a new alien language which uses the latin alphabet. However, the order among letters are unknown to you. You receive a list of non-empty words from the dictionary, where words are sorted lexicographically by the rules of this new language. Derive the order of letters in this language. [Leetcode](https://leetcode.com/problems/alien-dictionary/)

```
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        adjc_list = collections.defaultdict(set)
        indegree = {c: 0 for word in words for c in word}
        
        N = len(words)
        
        for i in range(N-1):
            j = 0
            while j < len(words[i]) and j < len(words[i+1]) and words[i][j] == words[i+1][j]:
                j += 1
            
            if j < len(words[i]) and j >= len(words[i+1]):
                return ""
            
            if j < len(words[i]):
                adjc_list[words[i][j]].add(words[i+1][j])
                indegree[words[i+1][j]] += 1

                
        output = []
        q = collections.deque([c for c in indegree if indegree[c] == 0])
        
        while q:
            c = q.popleft()
            output.append(c)
            
            for nei in adjc_list[c]:
                indegree[nei] -= 1
                
                if indegree[nei] == 0:
                    q.append(nei)
        
        return "".join(output)
```


# References
**Great explanation on how to tackle K-min problems:** https://leetcode.com/problems/k-th-smallest-prime-fraction/discuss/115819/Summary-of-solutions-for-problems-%22reducible%22-to-LeetCode-378
