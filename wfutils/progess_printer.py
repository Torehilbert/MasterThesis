import math


class ProgressPrinter():
    def __init__(self, steps, header="Epoch", sign_start=" ", sign_tick="-", sign_end=">", evolution_resolution=10, newline_at_end=True, print_evolution_number=True):
        self.total_steps = steps
        self.header = header
        self.sign_start = sign_start
        self.sign_end = sign_end
        self.sign_tick = sign_tick
        self.print_interval = 1.0 / evolution_resolution
        self.flag_new_line = newline_at_end
        self.flag_print_number = print_evolution_number

        self.evolutions = 0
        self.progress_counter = 0
        self.num_prints = 0

    def step(self):
        self.progress_counter += 1
        progress = self.progress_counter / self.total_steps
        
        target_num_prints = round(math.floor(progress / self.print_interval))
        while(self.num_prints < target_num_prints):
            print(self.sign_tick, end="", flush=True)
            self.num_prints += 1
        
        if(self.progress_counter == self.total_steps):
            print(self.sign_end, end="\n" if self.flag_new_line else "", flush=True)

    def start(self):
        self.evolutions += 1
        self.progress_counter = 0
        self.num_prints = 0

        str_ev_count = (" %3d" % (self.evolutions)) if self.flag_print_number else ""
        print(self.header + str_ev_count + self.sign_start, end="", flush=True)

    

if __name__ == "__main__":
    train_printer = ProgressPrinter(500000, newline_at_end=False)
    val_printer = ProgressPrinter(300000, header="", print_evolution_number=False)

    for epoch in range(12):
        train_printer.start()
        for i in range(500000):
            train_printer.step()

        val_printer.start()
        for i in range(300000):
            val_printer.step()
    
    
    