import curses
import main_program
import os



def option_one(stdscr):
    main_program.argazkiei_buelta_eman()
    stdscr.clear()
    stdscr.addstr(0, 0, "Argazkiei buelta emanda. Zapaldu edozein tekla jarraitzeko...")
    stdscr.refresh()
    stdscr.getch()

def option_two(stdscr):
    main_program.buelta_emandako_predikzioak_margoztu()
    stdscr.clear()
    stdscr.addstr(0, 0, "Predikzioak margoztu dira")
    stdscr.refresh()
    stdscr.getch()

def option_three(stdscr):
    main_program.buelta_emandako_predikzioak_margoztu_obb()
    stdscr.clear()
    stdscr.addstr(0, 0, "OBB Predikzioak margoztu dira")
    stdscr.refresh()
    stdscr.getch()

def option_four(stdscr):
    main_program.buelta_emandako_predikzioak_esportatu()
    stdscr.clear()
    stdscr.addstr(0, 0, "Predikzioak margoztu dira")
    stdscr.refresh()
    stdscr.getch()

def option_five(stdscr):
    main_program.buelta_emandako_predikzioak_esportatu_obb()
    stdscr.clear()
    stdscr.addstr(0, 0, "Predikzioak margoztu dira")
    stdscr.refresh()
    stdscr.getch()

def exit_program(stdscr):
    stdscr.clear()
    stdscr.addstr(0, 0, "Exiting the program...")
    stdscr.refresh()
    stdscr.getch()

def show_menu(stdscr):
    stdscr.clear()  # Clear screen to avoid refreshing issues
    stdscr.addstr(0, 0, "--- Menu Nagusia ---")
    stdscr.addstr(1, 0, "1. Sarrerako argazkieki buelta eman")
    stdscr.addstr(2, 0, "2. Predikzioak margoztu")
    stdscr.addstr(3, 0, "3. OBB predikzioak margoztu")
    stdscr.addstr(4, 0, "4. Predikzioak dataframean itzuli")
    stdscr.addstr(5, 0, "5. Predikzioak dataframean itzuli OBB")
    stdscr.addstr(6, 0, "6. Exit")
    stdscr.refresh()

def main(stdscr):
    # Initial setup
    curses.curs_set(0)  # Hide the cursor
    stdscr.nodelay(0)  # Block until user presses a key (blocking input)
    
    menu_options = {
        "1": option_one,
        "2": option_two,
        "3": option_three,
        "4": option_four,
        "5": option_five,
        "6": exit_program
    }

    while True:
        show_menu(stdscr)  # Show the menu

        choice = stdscr.getch()  # Wait for a key press (blocking)

        # Check if the input is a valid menu option
        if choice == ord('1'):
            menu_options['1'](stdscr)
        elif choice == ord('2'):
            menu_options['2'](stdscr)
        elif choice == ord('3'):
            menu_options['3'](stdscr)
        elif choice == ord('4'):
            menu_options['4'](stdscr)
        elif choice == ord('5'):
            menu_options['5'](stdscr)
        elif choice == ord('6'):
            menu_options['6'](stdscr)
            break  # Exit the program after 'Exit' option is selected
        else:
            stdscr.clear()
            stdscr.addstr(0, 0, "Invalid option, try again.")
            stdscr.refresh()
            stdscr.getch()  # Wait for user input to clear the error message

# Run the curses application
if __name__ == "__main__":
    curses.wrapper(main)
