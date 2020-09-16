summary: How to Write a Codelab
id: how-to-write-a-codelab
categories: Sample
tags: first
status: Published 
authors: Vlad
Feedback Link: https://vtoie.com

# TensorFlow 1 : The "Hello World!" of Machine Learning
## Introduction 
Duration: 6

### What You’ll Learn 
In this codelab you'll learn the basic "Hello World" of machine learning where, instead of programming explicit rules in a language such as Java or C++, you'll build a system that is trained on data to infer the rules that determine a relationship between numbers.

Consider the following problem: You're building a system that performs activity recognition for fitness tracking. You might have access to the speed at which a person is moving, and attempt to infer their activity based on this speed using a conditional:
![](assets/1.1.png)
```python
if speed < 4: 
  status = WALKING
```
You could extend this to running with another condition:
![](assets/1.2.png)
```python
if speed < 4: 
  status = WALKING
else:
  status = RUNNING
```
In a final condition you could similarly detect cycling:
![](assets/1.3.png)
```python
if speed < 4: 
  status = WALKING
else if speed < 12:
  status = RUNNING
else
  status = BIKING
```
Now consider what happens when you want to include an activity like golf? Suddenly it's less obvious how to create a rule to determine the activity.
![](assets/1.4.png")
```python
# Now what?? :(
```
It's extremely difficult to write a program (expressed in code) that will give us the golfing activity. So what do you do? That's where machine learning can be used to solve the problem!
<!-- ------------------------ -->
## Setting Duration
Duration: 2

To indicate how long each slide will take to go through, set the `Duration` under each Heading 2 (i.e. `##`) to an integer. 
The integers refer to minutes. If you set `Duration: 4` then a particular slide will take 4 minutes to complete. 

The total time will automatically be calculated for you and will be displayed on the codelab once you create it. 

<!-- ------------------------ -->
## Code Snippets
Duration: 3

To include code snippets you can do a few things. 
- Inline highlighting can be done using the tiny tick mark on your keyboard: "`"
- Embedded code

### JavaScript

```javascript
{ 
  key1: "string", 
  key2: integer,
  key3: "string"
}
```

### Java

```java
for (statement 1; statement 2; statement 3) {
  // code block to be executed
}
```

<!-- ------------------------ -->
## Hyperlinking and Embedded Images
Duration: 1### Hyperlinking
[Youtube - Halsey Playlists](https://www.youtube.com/user/iamhalsey/playlists)

### Images
![alt-text-here](assets/backrooms.jpg)

<!-- ------------------------ -->
## Other Stuff
Duration: 1

Checkout the official documentation here: [Codelab Formatting Guide](https://github.com/googlecodelabs/tools/blob/master/FORMAT-GUIDE.md)
