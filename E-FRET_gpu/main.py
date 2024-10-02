import turtle

# 设置窗口
window = turtle.Screen()
window.bgcolor("white")

# 创建画笔
pen = turtle.Turtle()
pen.speed(0)
pen.color("red")


# 绘制爱心
def draw_heart():
    pen.penup()
    pen.goto(0, -100)
    pen.pendown()
    pen.begin_fill()
    pen.left(140)
    pen.forward(111.65)
    for _ in range(200):
        pen.right(1)
        pen.forward(1)
    pen.left(120)
    for _ in range(200):
        pen.right(1)
        pen.forward(1)
    pen.forward(111.65)
    pen.end_fill()


# 动画循环
while True:
    pen.clear()
    draw_heart()
    window.update()
