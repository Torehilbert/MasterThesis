


if __name__ == "__main__":
    PATH1 = r"D:\Speciale\Code\output\Performance Trainings\C48\C48_Run1\series.txt"
    PATH2 = r"D:\Speciale\Code\output\Performance Trainings\C48\2020-10-21--18-53-25_Training_7793\series.txt"

    f1 = open(PATH1, 'r')
    f2 = open(PATH2, 'r')

    for i in range(10):
        l1 = f1.readline()
        l2 = f2.readline()
        if(l1 != l2):
            print("NO")
            exit(0)
    
    print("YES")