---
title: Dynamic Programming
date: 2020-05-14 21:00:00
comments: true
author: Yi-Wei
categories:
- nlp study group
tags:
- Leetcode
---
###### tags: `study`
# Dynamic Programming

## 動態規劃 Dynamic Programming : 
idea : 動態規劃(Dynamic Programming)是指將一個較大的問題定義為較小的子問題組合，先處理較小的問題並將結果儲存起來(通常使用表格)，再進一步以較小問題的解逐步建構出較大問題的解。
<!-- more -->

Easy solution in Fibonacci Squence or Leetcode(70 :Climbing Stairs.)


![](https://i.imgur.com/RUsdK9G.png)

- First we can do it in recusive part, but no memory save.
![](https://i.imgur.com/nN5t6ja.png)
The time limit exceeded! ![](https://i.imgur.com/dqJtZEc.png)
Because we do more time in the repeated work, and the Time Complexity is $O(2^n)$.

1. we can do in the memory part, recursive.
![](https://i.imgur.com/b1UDhjA.png)
Time Complexity is O(n).
2. we can do the DP(Dynamic programming).
![](https://i.imgur.com/tthxRWq.png)

## 經典動態規劃問題
- Shortest path problem in weighted directed graph (negative edge allowed but no negative cycles)

![](https://i.imgur.com/8oSmiVh.gif)
Question : Find the minimum path in th graph
approch : (forward or backward)
Here is the forward DP
let $f(k)$ is the shortest path to the k point.
$f(k) = 0$
$W(k-1,k)$ is the distance from $k-1$ to $k$
recursive with $f(k-1) = min\{f(k) + W(k-1,k)\}$

Time Complexity : O(|V|+|E|)

## The Longest Common Subsequence (LCS) Problem:
Question : Give two Sequences $X = <x_1,x_2,....x_n>, Y = <y_1,y_2,...y_m>$ find the Longest common subsequence(words not need to consecutive)
solution :![](https://i.imgur.com/JIsEYh9.png)
example for LCS![](https://i.imgur.com/Em8IwMl.png)


