## Деревья Штейнера

Решение задачи о нахождении локально минимального дерева посредством алгоритма Мелзака-Хванга

Автор реализации: Михаил Кучеренко,  
МГТУ им. Н.Э. Баумана, ИУ5-64, 2019г.

### Задание

![png](https://s247myt.storage.yandex.net/rdisk/637ea33f18fc1e5a52cdd0b780e47b87f7c9a266336662ce747a5a5347fed0c6/5cf48006/HDMHEPhyzZhO2v3_qfpXyt_GxZl5IQOoCB48DGXivWPy4jOI94EvBVpmaPrlI5mlQMbRITHgCTOg-s6yDBUwsA==?uid=0&filename=%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202019-06-03%20%D0%B2%201.02.36.png&disposition=inline&hash=&limit=0&content_type=image%2Fpng&fsize=320991&hid=e3fd287000424dca22aaf8fc90369427&media_type=image&tknv=v2&etag=8e4503283f986f9ad1166b5ea6fc1782&rtoken=Tm48EaYHnqaR&force_default=no&ycrid=na-b1e63759b640a0c0e3a788bafa0dee4c-downloader23h&ts=58a61c67b8d80&s=eeb1075a5771a3981270a8c73cdcf5bc5ab5668e86f6d2335746af49f30a707d&pb=U2FsdGVkX19T8Z84jQQoPL0ap7b4W3dASSyVL9c-ID9d9I-13tdO1mfUtX8H0DrwKnD32zbGhd7FRqBqTAzttsf3qCEX1vhDIU-NkyJVsUs)


### Применение

```python
# Номер варианта
d = 9

# Списки смежности из условия
G1 = [[7],[7],[8],[8],[9],[9],[1,2,10],[3,4,10],[5,6,10],[7,8,9]]
G2 = [[7],[7],[8],[10],[10],[9],[1,2,8],[3,7,9],[6,8,10],[4,5,9]]

# Вычисление вектора L по номеру варианта
l = [None] * 6
for i in reversed(range(1,7,1)):
    n = d % i + 1
    m = 7 - i
    print '{}. {} mod {} + 1 = {}'.format(m, d, i, n)
    for j in range(len(l)):
        if l[j] is None:
            n -= 1
        if n == 0:
            l[j] = m
            break
    print '   L = {}\n'.format(l)
# Для удобства рассчетов точек P1-P6:
l = [None] + l

# Решение задачи
solve(l, G1)
solve(l, G2)
```
