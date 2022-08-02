"""
Collection of functions and classes that may be useful in other scripts.

    - Functions:
        - `PrintThis`:  Prints a list of all functions and classes in this module.
        - `clear`:  Clears the terminal screen.
        - `println`:  Prints a newline following the object passed. Shortcut for `print(obj, end='\\n')`.
        - `perf`:  Prints the performance of a given function (execution time, peak memory use).
        - `delay`:  Much better alternative to `time.sleep()`.
        - `printdashes`:  Prints a line of dashes with specified length.
        - `dash`:  Returns a string with the specified number of dashes.
        - `any_match`:  Checks a string for any matches defined by a set of conditions.
        - `convert_to_local_time`:  Converts a datetime object to a new one in the local timezone.
        - `sort`:  Sorts a string, tuple, list, or dictionary.
        - `reverse`:  Reverses the order of a string, tuple, range, or list.
        - `view`:  Uses the `pyjsonviewer` module to view a dictionary or JSON object.
        - `fmt`:  Formats a number with (optional)commas and specified float precision.
        - `exctract_digits`:  Extracts digits from a string.
        - `weighted_average`:  Calculates the weighted average of a list of numbers and weights.
        - `print_matrix`:  Prints a given matrix in a nice format.
        - `print_array`:  Prints an array in a nice format.
        - `wolfram`:  Returns Wolfram Alpha's response/result of a given query.
        - `char2key`:  Converts a character to a virtual-key-code.
        - `string2keys`:  Converts a string to a list of virtual-key-codes.
        - `kill_process`:  Kills a Windows process with the given name, if it exists.
        - `show_processes`:  Prints all processes currently running on the PC.
        - `get_window`:  Returns `WindowManager` objects of the given windows, for accessing the `WindowManager` class methods.
        - `intermediate_points`:  Returns a list of intermediate points between start and end, via Bresenham's line algorithm.
        - `dict_to_struct`:  Converts a Python dictionary to the format of a C++ struct.
        - `scrape_wikipedia_table`:  Scrapes a wikipedia table from a given URL using the BeautifulSoup library.

    - Classes:
        - `Animate`:  A class that animates a given function in two dimensions (x,y).
        - `Plot`:  A `pyplot` wrapper class for plotting data easily.
        - `Plot3D`:  A `mpl_toolkits.mplot3d` wrapper class for plotting data in three dimensions.
        - `Point`:  A class for representing points and vectors in 2D space.
        - `System`:  A class for interacting with the Windows OS.
        - `Web3`:  A wrapper for the `web3` module.
        - `File`:  A class to handle files and the manipulation of them, including PDFs.
        - `Window`:  A class for managing application/program windows.
        - `Keyboard`:  A class for interfacing with keyboard library.
        - `Screen`:  A class for getting various information from the screen.
        - `Math`:  A class containing several math functions.
        - `SolarIrradiance`:  A class for getting current solar irradiance data.
        - `Git`:  A class containing several Git functions.
        - `Regex`:  A class for simplifying the matching of strings against regular expressions.
"""




def PrintThis():
    """
    Prints a list of all functions and classes in this module, with their docstrings.
    """
    clear()
    functions = [f for f in globals().values() if callable(f) and f.__name__ != 'Print']          # Get all functions in this module excluding this one.
    for f in functions:
        print('\n'+'-'*(len(max(f.__doc__.split('\n'), key=len))))
        print('\n'+'{}()'.format(f.__name__))                                       # prints the function name
        print(f.__doc__)                                                            # prints the function's docstring
        print('-'*(len(max(f.__doc__.split('\n'), key=len))))
        if f == functions[-1]:  print('\n\n')
        else:  print('\n',end='')


def clear(newlines=2):
    """
    Clears the terminal screen.

    Args:
        `newlines` (int):  The number of newlines to print after clearing the screen. Default is 2.
    """
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    for i in range(newlines): print()

        
def println(obj):
    """
    Prints a newline following the object passed. Shortcut for `print(obj, end='\\n\\n')`.

    Args:
        `obj` (object):  The object to print.
    """
    print(obj, end='\n\n')

    
def perf(func):
    """
    Function wrapper that measures performance of a function.

    - Returns execution time in seconds,
    - memory usage in megabytes,
    - and peak memory usage in megabytes
    """
    import tracemalloc
    from functools import wraps
    from time import perf_counter

    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = perf_counter()
        func_return = func(*args, **kwargs)
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        if peak_memory/10**6 < 0.001:
            mem_unit = 'self._keyboard'
            peak_memory = peak_memory/10**3
        else:
            mem_unit = 'MB'
            peak_memory = peak_memory/10**6
        finish_time = perf_counter()
        execution_time = finish_time - start_time
        if execution_time < 0.001:
            time_unit = 'ms'
            execution_time = execution_time*1000
        else: time_unit = 's'
        print(f"\n\nFunction: {func.__name__}")
        print(f"Time elapsed:\t\t {execution_time:.6f} {time_unit}")
        print(f"Peak memory usage:\t {peak_memory:.6f} {mem_unit}")
        print('-'*38+'\n')
        tracemalloc.stop()
        return func_return
    return wrapper


def delay(duration):
    """
    Significantly better version of `time.sleep()`.
    Useful for small delays, on the order of milliseconds.\n
    To save on resources, pass in the duraction as seconds, to prevent relatively lengthy FLOPs.

    Args:
        `duration`: The amount of time to delay, in seconds. Note: 0.1 seconds = 100 milliseconds.
    
    Returns:
        `None`
    
    Summary of test results:
        - `duration = 1`
            - sleep function:  0.0194% error
            - delay function:  0.0003% error
        - `duration = 0.5`
            - sleep function:  0.0437% error
            - delay function:  0.0008% error
        - `duration = 0.1`
            - sleep function:  0.1686% error
            - delay function:  0.0061% error
        - `duration = 0.01`
            - sleep function:  2.1366% error
            - delay function:  0.2041% error
        - `duration = 0.001`
            - sleep function:  10.879% error
            - delay function:  0.1188% error
        - `duration = 0.0001`
            - sleep function:  829.26% error
            - delay function:  1.2298% error
    """
    from time import perf_counter
    start = perf_counter()
    while perf_counter() - start < duration:
        pass
    
    
def printdashes(length=50, char='-', endtype='\n'):
    """
    Prints a line of dashes with specified length.

    Args:
        `length` (int):  The length of the line.  Default is 50.
        `char` (str):  The character to print.  Default is  `-`.
        `endtype` (str):  The type of end character to print.  Default is  `\\n`.
    """
    print(char*length, end=endtype)

    
def dash(length):
    """
    Returns a line of dashes with specified length.

    Args:
        `length` (int):  The length of the line.
    
    Returns:
        `str` (str):  The line of dashes.
    """
    return '-'*length


def any_match(match_conditions, string, case_sensitive=False):
    """
    Checks a string for any matches defined by a set of conditions.

    Args:
        `match_conditions` (list or tuple, str):  The set of conditions that should evaluate as `True`.
        `string` (str):  The string to check for the existence of these conditions.
        `case_sensitive` (bool):  Whether or not to consider case-sensitive matches only.  Default is `False`.

    Returns:
        `bool` (bool):  `True` if any of the conditions are in the given string.
    
    Examples:
        >>> any_match(['a', 'b', 'c'], 'abc')
        True
        >>> any_match(("rooted", "snared"), 'You are rooted!')
        True
        >>> any_match(("rooted", "snared"), 'You are Snared!')
        True if not case_sensitive, else False
    """
    if not case_sensitive:
        string = string.lower()
        match_conditions = [condition.lower() for condition in match_conditions]
    return True if any(condition in string for condition in match_conditions) else False


def convert_to_local_time(datetime_obj, fmt=None):
    """
    Converts a datetime object to the local timezone.

    Args:
        `datetime_obj` (datetime):  The datetime object to convert.
        `fmt` (str):  The format to return the datetime object in.  Default is `None` (will return datetime object).

    Returns:
        `datetime` (datetime):  The datetime object in the local timezone.
        `str` (str):  The datetime object in the local timezone, in the format specified by `fmt`, if provided.
    """
    import pytz
    from datetime import datetime
    local_tz = pytz.timezone('US/Central')
    local_dt = datetime_obj.astimezone(local_tz)
    if fmt is None: return local_dt
    else: return local_dt.strftime(fmt)


# TODO: fix the sorting aspect of this function
def sort(obj, ascending=True, key=None, index_list=False):
    """
    Sorts a string, tuple, list, or dictionary.

    Args:
        `obj` (str, tuple, list, dict):  The object to sort.
        `ascending` (bool):  Whether or not to sort in ascending order. Default = True.
        `key` (str):  The key to sort by. Default = None.
        `index_list` (bool):  Whether or not to return the sorted indices. Default = False.
    
    Returns:
        `sorted_obj` (str, tuple, list, dict):  The sorted object.
        `index_list` (list):  The sorted indices, if `index_list` is True.
    """
    if key is None:
        if index_list:
            return sorted(obj, key=None, reverse=not ascending), sorted(range(len(obj)), key=None, reverse=not ascending)
        else:
            return sorted(obj, key=None, reverse=not ascending)
    else:
        if index_list:
            return sorted(obj, key=key, reverse=not ascending), sorted(range(len(obj)), key=key, reverse=not ascending)
        else:
            return sorted(obj, key=key, reverse=not ascending)
    

def reverse(obj):
    """
    Reverses the order of a string, tuple, range, or list.

    Args:
        `obj` (str, tuple, range, list):  The object to reverse.
    
    Returns:
        `list` (list):  The reversed order.
    """
    return list(reversed(obj))


def view(jsonobj):
    """
    Uses the `pyjsonviewer` module to view a dictionary or JSON object.

    Args:
        `jsonobj` (dict):  The JSON object to view.
    
    Returns:
        `None`
    """
    from pyjsonviewer import view_data
    view_data(json_data=jsonobj)


def fmt(num, precision=2, commas=True, _return=False):
    """
    Formats a number with commas and specified float precision.

    Args:
        `num` (int/float):  The number to print or format.
        `precision` (int):  The float precision to use. Default = 2.
        `_commas` (bool):  Whether or not to use commas. Default = True.
        `_return` (bool):  Whether or not to return the string. Default = False (just prints it).
    
    Returns:
        `num` (str):  The number with commas and specified float precision, if `_return` is True.  Else:  `None`.
    """
    if _return:
        if commas:  return '{:,.{}f}'.format(num, precision)
        else:  return '{:.{}f}'.format(num, precision)
    else:
        if commas:  print('{:,.{}f}'.format(num, precision))
        else:  print('{:.{}f}'.format(num, precision))


def extract_digits(string):
    """
    Extracts all digits from a string.

    Args:
        `string` (str):  The string to extract digits from.
    
    Returns:
        `digits` (str):  The digits in the string.
    """
    return ''.join(filter(str.isdigit, string))


def weighted_average(num_list, weight_list):
    """
    Calculates the weighted average of a list of numbers and weights.

    Args:
        `num_list` (list of lists):  The numbers to average, one list for each weight.
        `weight_list` (list):  The list of weights to use.
    
    Returns:
        `average` (float):  The weighted average.
    
    Example:
        >>> weighted_average([[0, 69], [0, 37], [0, 94], [0, 72]], [0.2, 0.3, 0.4, 0.1])
        >>> 34.85
    """
    avg_list = [sum(num_list[i])/len(num_list[i]) for i in range(len(num_list))]
    return sum([avg_list[i]*weight_list[i] for i in range(len(avg_list))])



def print_matrix(mat, precision=2, commas=True):
    """
    Prints a given matrix in a nice format.

    Args:
        `mat` (list, numpy.array, numpy.matrix):  The matrix to print.
        `precision` (int):  The float precision to use. Default = 2.
        `commas` (bool):  Whether or not to use commas. Default = True.
    
    Returns:
        `None`
    
    Example forms for `mat`:
        >>> [ [1,2,3],[4,5,6],[7,8,9] ]
        >>> numpy.array([ [1,2,3],[4,5,6],[7,8,9] ])
        >>> numpy.matrix([ [1,2,3],[4,5,6],[7,8,9] ])
    """
    from numpy import matrix
    if isinstance(mat, matrix):  mat = mat.tolist()
    widths = [max([len(fmt(x,precision,commas,_return=True)) for x in row]) for row in mat]
    print('\n'.join(['      '.join([fmt(x,precision,commas,_return=True).rjust(w) for x,w in zip(row,widths)]) for row in mat]))


def print_array(arr, precision=2, commas=True):
    """
    Prints an array in a nice format.

    Args:
        `arr` (list, numpy.array):  The array to print.
        `precision` (int):  The float precision to use. Default = 2.  Pass 'max' to use the precision of the highest-precision number in the array.
        `commas` (bool):  Whether or not to use commas. Default = True.
    
    Returns:
        `None`
    """
    if precision == 'max': precision = max( len(str(x).split('.')[1]) for x in arr if '.' in str(x) )
    print('    '.join([fmt(x,precision,commas,_return=True) for x in arr]))


def wolfram(query, only_result=True, only_print=False):
    """
    Queries Wolfram Alpha with a query string.

    Args:
        `query` (str):  The query string.
        `only_result` (bool):  Whether to return only the result of the query. Default = True.
            -> If `only_result` is False, all contents of the `pods` are returned instead.
        `only_print` (bool):  Whether to print the result of the query instead of returning. Default = False.
    
    Returns:
        `result` (str):  The result of the query.
    """
    import wolframalpha
    APP_ID = 'xxxxx'
    client = wolframalpha.Client(APP_ID)
    if only_result:
        result = client.query(query)
        if only_print: print(f"\n{next(result.results).text}\n")
        else: return next(result.results).text
    else:
        result = client.query(query)
        for pod in result.pods:
            for sub in pod.subpods:
                print(sub.plaintext)
                print('-'*(len(max(sub.plaintext.split('\n'), key=len))))
        if only_print:
            print('')
            [print(sub.plaintext) for pod in result.pods for sub in pod.subpods]
            print('')
        else:
            return [sub.plaintext for pod in result.pods for sub in pod.subpods]



def char2key(c):
    """
    Converts a character to a virtual-key-code.
    
    Args:
        `c`:  The character to convert.
    
    Returns:
        `vk_key`:  The virtual-key-code.
    """
    from ctypes import windll
    result = windll.User32.VkKeyScanW(ord(c))
    shift_state = (result & 0xFF00) >> 8
    vk_key = result & 0xFF
    print(f"{c} -> {vk_key}")
    return vk_key


def string2keys(string):
    """
    Converts a string to a list of virtual-keys-codes.

    Args:
        `string`:  The string to convert.
    
    Returns:
        A list of virtual-keys-codes.
    """
    return [char2key(c) for c in string]


def kill_process(name):
    """
    Kills the process with the given name, if it exists.

    Args:
        `name` (str):  The name of the process to kill.
    
    Returns:
        `None`
    """
    import psutil
    for proc in psutil.process_iter():
        if name in proc.name():
            try:
                proc.kill()
                print(f"\n> Process \"{proc.name()}\" was killed successfully.\n")
            # except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess): pass
            except: print(f"\n> Process \"{name}\" either couldn't be killed or was not found.\n")

                
def show_processes():
    """
    Prints all processes currently running.
    """
    import os
    for line in os.popen('tasklist').read().splitlines(): print(line)


def get_window(window_name):
    """
    Gets a window by its name.

    Args:
        `window_name` (str):  The name of the window.
    
    Returns:
        `window` (Window instance):  The window.
    """
    window = Window()
    window.find_window_wildcard(f".*{window_name}.*")
    return window


def intermediate_points(start, end):
    """
    Returns a list of intermediate points between start and end via Bresenham's line algorithm.

    Args:
        `start` (tuple):  The start point.
        `end` (tuple):  The end point.
    
    Returns:
        `points_in_line` (list, tuples):  The list of intermediate points.
    """
    points_in_line = []
    x0 = start[0]; y0 = start[1]
    x1 = end[0];   y1 = end[1]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points_in_line.append((x,y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points_in_line.append((x,y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points_in_line.append((x,y))
    return points_in_line


def dict_to_struct(dictionary, struct_name):
    """
    Converts a Python dictionary to the format of a C++ struct.  
     Returns nothing, and instead prints the struct to the console.

    Args:
        `dictionary` (dict):  The dictionary to convert.
        `struct_name` (str):  The name of the struct.
    """
    print("\n--------------------\n")
    print(f"struct {struct_name}")
    print("{")
    statements = []
    for key,value in dictionary.items():
        if " " in key: key = key.replace(" ","_")
        if "-" in key: key = key.replace("-","_")
        if value is None:
            value = "\"None\""
            key = "std::string "+key                                    # C++ file must include <string>
        elif isinstance(value,int):
            key = "int "+key
        elif isinstance(value,str):
            value = '\"'+value+'\"'
            key = "std::string "+key                                    # C++ file must include <string>
        elif isinstance(value,float):
            if "e+" in str(value):
                key = "double "+key
                val = str(value).split("e+")[0]
                exp = int(str(value).split("e+")[1])
                value = val + " * pow(10," + str(exp) + ")"             # C++ file must include <cmath>
            elif "e-" in str(value):
                key = "double "+key
                val = str(value).split("e-")[0]
                exp = int(str(value).split("e-")[1])
                value = val + " * pow(10," + str(exp) + ")"             # C++ file must include <cmath>
            elif str(value) == "inf" or str(value) == "math.inf":
                key = "double "+key
                value = "std::numeric_limits<double>::infinity()"       # C++ file must include <limits>
            else: key = "float " +key
        elif isinstance(value,bool):
            if value: key = "true " +key
            else:     key = "false "+key
        elif isinstance(value,dict):
            key = "struct " +key
            value = "CHANGE THIS!!!!"
        statements.append(f"    {key} = {value};")
    print("\n".join(statements))
    print('};')
    print("\n--------------------\n")


def scrape_wikipedia_table(url,table_headings,columns_to_keep,txt_file=False):
    """
    Scrapes a Wikipedia table from a given URL using `BeautifulSoup`.

    Args:
        `url` (str):  The URL of the table.
        `table_headings` (list):  The headings of the table to scrape (for identifying the correct table on the page to scrape).
        `columns_to_keep` (list):  The columns to keep (their names/headings) from the table (can be less or more than the number of headers in `table_headings`).
        `txt_file` (bool):  Whether to save the table to a text file.  Default is False.
    
    Returns:
        `table` (dict):  The table in dictionary form, with row number (starting from 0) as the keys, and column names & data as values.
    
    Example:
        >>> table = scrape_wikipedia_table(
                url="https://en.wikipedia.org/wiki/Synchronous_orbit",
                table_headings= ["Orbit", "Body\'s Mass (kg)", "Sidereal Rotation period", "Semi-major axis (km)", "Altitude"],
                columns_to_keep=["Orbit", "Body\'s Mass (kg)", "Sidereal Rotation period", "Semi-major axis (km)"],
            )
        >>> {
                0:
                    Orbit: Geostationary orbit (Earth)
                    Body Mass (kg): 5.97237e24
                    Sidereal Rotation period: 0.99726968 d
                    Semi-major axis (km): 42,164 km (26,199 mi)
                1:
                    Orbit: areostationary orbit (Mars)
                    Body Mass (kg): 6.4171e23
                    Sidereal Rotation period: 88,642 s
                    Semi-major axis (km): 20,428 km (12,693 mi)
                2:
                    Orbit: Ceres stationary orbit
                    Body Mass (kg): 9.3835e20
                    Sidereal Rotation period: 9.074170 h
                    Semi-major axis (km): 1,192 km (741 mi)
            }
    """
    import urllib.request
    from bs4 import BeautifulSoup
    req = urllib.request.urlopen(url)
    article = req.read().decode()
    with open(url.split("wiki/")[1]+".html", 'w', encoding='unicode_escape') as fo:
        fo.write(article)
    # Load article, turn into soup and get the <table>s.
    article = open(url.split("wiki/")[1]+".html").read()
    soup = BeautifulSoup(article, 'html.parser')
    tables = soup.find_all('table', class_='sortable')
    # Search through the tables for the one with the headings we want.
    for table in tables:
        ths = table.find_all('th')
        headings = [th.text.strip() for th in ths]
        if headings[:len(table_headings)] == table_headings:
            break
    # Extract the columns we want and write to a semicolon-delimited text file.
    with open('table.txt', 'w') as fo:
        for tr in table.find_all('tr'):
            tds = tr.find_all('td')
            if not tds:
                continue
            if len(columns_to_keep) == 2:
                a, b = [td.text.strip() for td in tds[:len(columns_to_keep)]]
                if "\\xd710" in a: a = a.replace("\\xd710","e")
                if "\\xd710" in b: b = b.replace("\\xd710","e")
                print('; '.join([a, b]), file=fo)
            elif len(columns_to_keep) == 3:
                a, b, c = [td.text.strip() for td in tds[:len(columns_to_keep)]]
                if "\\xd710" in a: a = a.replace("\\xd710","e")
                if "\\xd710" in b: b = b.replace("\\xd710","e")
                if "\\xd710" in c: c = c.replace("\\xd710","e")
                print('; '.join([a, b, c]), file=fo)
            elif len(columns_to_keep) == 4:
                a, b, c, d = [td.text.strip() for td in tds[:len(columns_to_keep)]]
                if "\\xd710" in a: a = a.replace("\\xd710","e")
                if "\\xd710" in b: b = b.replace("\\xd710","e")
                if "\\xd710" in c: c = c.replace("\\xd710","e")
                if "\\xd710" in d: d = d.replace("\\xd710","e")
                print('; '.join([a, b, c, d]), file=fo)
            elif len(columns_to_keep) == 5:
                a, b, c, d, e = [td.text.strip() for td in tds[:len(columns_to_keep)]]
                if "\\xd710" in a: a = a.replace("\\xd710","e")
                if "\\xd710" in b: b = b.replace("\\xd710","e")
                if "\\xd710" in c: c = c.replace("\\xd710","e")
                if "\\xd710" in d: d = d.replace("\\xd710","e")
                if "\\xd710" in e: e = e.replace("\\xd710","e")
                print('; '.join([a, b, c, d, e]), file=fo)
    # Read the table.txt file and create a dictionary of the columns_to_keep_vars and the corresponding values.
    with open('table.txt', 'r') as fo:
        table = [line.strip().split(';') for line in fo]
    table = [dict(zip(columns_to_keep, row)) for row in table]

    if not txt_file:
        # delete the table.txt file
        import os
        os.remove('table.txt')
    return table








class Animate:
    """
    A class that animates a given function in two dimensions (x, y).
    """
    def __init__(self, func, min, max, start, end, steps):
        """
        Args:
            `func` (function):  The function to animate, aka what generates `y` values (function of `x` and `i`).
            `min` (int):  The minimum value for `func`.
            `max` (int):  The maximum value for `func`.
            `start` (int):  The start value, aka the initial `x` value.
            `end` (int):  The end value, aka the final `x` value.
            `steps` (int):  The number of steps to take.
        
        Example:
            >>> def func(x, i): return np.sin(2 * np.pi * (x - 0.01 * i))
            >>> anim = Animate(func, min=-2, max=2, start=0, end=4, steps=1000)
            >>> anim.run()
            >>> anim.save('sin.gif')
        """
        import numpy as np
        from matplotlib import pyplot as plt
        plt.style.use('dark_background')
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(start, end), ylim=(min, max))
        self.func = func
        self.x = np.linspace(start, end, steps)
        self.line, = self.ax.plot([], [], lw=3)

    
    def init(self):
        """
        Initializes the animation.
        """
        # creating an empty plot/frame (canvas) to plot on
        self.line.set_data([], [])
        return self.line,
    
    def animate(self, i):
        """
        Animates the function.
        """
        self.line.set_data(self.x, self.func(self.x, i))
        return self.line,
    
    def run(self, frames=200, interval=20):
        """
        Runs the animation.

        Args:
            `frames` (int):  The number of frames to animate.
            `interval` (int):  The interval between frames.
        """
        from matplotlib import pyplot as plt
        from matplotlib.animation import FuncAnimation
        self.anim = FuncAnimation(self.fig, self.animate, init_func=self.init, frames=frames, interval=interval, blit=True)
        plt.show()
    
    def save(self, filename):
        """
        Saves the animation as a gif.
        """
        if filename[-4:] != '.gif': filename += '.gif'
        self.anim.save(filename, writer='imagemagick')






class Plot:
    """
    A `pyplot` wrapper class for plotting data.
    Constructor takes four arguments:
        - `size` (tuple):  The size of the plot, in pixels. Default = (900,600).
        - `whitespace (bool)`:  Whether to keep whitespace around the plot. Default = True. `whitespace = False` also gets rid of the axes.
        - `xbounds` (tuple):  The x-axis bounds. Default = None.
        - `ybounds` (tuple):  The y-axis bounds. Default = None.
    """
    def __init__(self, size=(900,600), whitespace=True):
        from matplotlib import pyplot as plt
        self.fig = plt.figure(figsize=(size[0]/96, size[1]/96), dpi=96)
        if not whitespace: self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax = self.fig.add_subplot(111)
        self.xlabel = 'x'
        self.ylabel = 'y'
        self._title = ''
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self._title)
        self._colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w','r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
        self._colors_no_white = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k']
        self._numlines = 0
    @property
    def colors(self): return self._colors[:7]
    @property
    def styles(self): return "\n-   solid\n--  dashed\n-.  dashdot\n:.  dotted\n"
    def set_numlines(self, numlines):
        self._numlines = numlines
    def x_axis(self, x):
        """
        Sets the x-axis bounds.
        
        Args:
            `x` (tuple):  The x-axis bounds (xmin, xmax).
        """
        self.ax.set_xlim(x)
    def y_axis(self, y):
        """
        Sets the y-axis bounds.
        
        Args:
            `y` (tuple):  The y-axis bounds (ymin, ymax).
        """
        self.ax.set_ylim(y)

    def x_label(self, label):
        """
        Sets the x-axis label.
        
        Args:
            `label` (str):  The label of the x-axis.
        """
        self.xlabel = label
        self.ax.set_xlabel(label)
    def y_label(self, label):
        """
        Sets the y-axis label.
        
        Args:
            `label` (str):  The label of the y-axis.
        """
        self.ylabel = label
        self.ax.set_ylabel(label)
    def title(self, title):
        """
        Sets the title of the plot.
        
        Args:
            `title` (str):  The title of the plot.
        """
        self._title = title
        self.ax.set_title(title)

    def line(self, x, y, color=None, style=None, linewidth=2, label=None):
        """
        Plots a line.
        
        Args:
            `x` (tuple):  The values defining the x-component of the line (x-initial, x-final).
            `y` (tuple):  The values defining the y-component of the line (y-initial, y-final).
            `color` (str):  The color of the line.  print(Plot.colors) to see a list of colors.  Default is `None`, which uses the next color in the list.
            `style` (str):  The style of the line.  print(Plot.styles) to see a list of styles.  Default is `None`, which uses the solid line linestyle.
            `linewidth` (int):  The width of the line. Default = 2.
            `label` (str):  The label of the line.  Default is `None`.
        """
        if color is None: color = self._colors[self._numlines]
        if style is None: style = '-'
        self.ax.plot(x, y, f"{color}{style}", label=label, linewidth=linewidth)
        self._numlines += 1
    
    def point(self, xy=None, x=None, y=None, color=None, marker='o', markersize=5, label=None):
        """
        Plots a point.
        
        Args:
            `xy` (tuple):  The x and y coordinates of the point.
            `x` (float):  The x-coordinate of the point.  Default is `None`, only specify if not using `xy` tuple argument.
            `y` (float):  The y-coordinate of the point.  Default is `None`, only specify if not using `xy` tuple argument.
            `color` (str):  The color of the point.  print(Plot.colors) to see a list of colors.  Default is `None`, which uses the next color in the list.
            `marker` (str):  The marker of the point.  Default is `o`.
            `markersize` (int):  The size of the marker. Default = 10.
            `label` (str):  The label of the point.  Default is `None`.
        """
        if color is None:
            try: color = self._colors[self._numlines]
            except IndexError: color = 'r'
        if xy is not None: x, y = xy
        self.ax.plot(x, y, f"{color}{marker}", markersize=markersize, label=label)
        self._numlines += 1
    
    def points(self, xy, color=None, marker='o', markersize=5, label=None):
        """
        Plots a series of points.
        
        Args:
            `xy` (list):  The x and y coordinates of the points.
            `color` (str):  The color of the points.  print(Plot.colors) to see a list of colors.  Default is `None`, which uses the next color in the list.
            `marker` (str):  The marker of the points.  Default is `o`.
            `markersize` (int):  The size of the markers. Default = 10.
            `label` (str):  The label of the points.  Default is `None`.
        """
        num_points = 0
        for x, y in xy:
            try: color = self._colors_no_white[num_points]
            except IndexError:
                num_points = 0
                color = self._colors_no_white[num_points]
            self.ax.plot(x, y, f"{color}{marker}", markersize=markersize, label=str(x) + ',' + str(y))
            num_points += 1
    
    def legend(self, loc='upper right'):
        """
        Adds a legend to the plot.
        
        Args:
            `loc` (str):  The location of the legend.  Default is `upper right`.
        
        Can be any of:
            `best`, `right`, `center`\n
            `upper right`, `upper left`
            `lower right`, `lower left`\n
            `center right`, `center left`
            `lower center`, `upper center`
        """
        self.ax.legend(loc=loc)
    
    # TODO: fix the vector function
    def vector(self, x, y, color=None, style=None, label=None):
        """
        Plots a vector.
        
        Args:
            `x` (tuple):  The values defining the x-component of the vector (x-initial, x-final).
            `y` (tuple):  The values defining the y-component of the vector (y-initial, y-final).
            `color` (str):  The color of the vector.  print(Plot.colors) to see a list of colors.  Default is `None`, which uses the next color in the list.
            `style` (str):  The style of the vector.  print(Plot.styles) to see a list of styles.  Default is `None`, which uses the solid line linestyle.
            `label` (str):  The label of the vector.  Default is `None`.
        """
        if color is None: color = self._colors[self._numlines]
        if style is None: style = '-'
        # self.ax.quiver(x[0], y[0], x[1]-x[0], y[1]-y[0], color=color, linestyle=style, label=label, linewidth=2)
        self._numlines += 1

    def show(self):
        """
        Shows the plot.
        """
        from matplotlib import pyplot as plt
        plt.show()
    
    def save(self):
        """
        Saves the plot to the current directory.
        """
        self.fig.savefig("plot.png")






# used for drawing vectors for the Plot3D class
from matplotlib.patches import FancyArrowPatch
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def do_3d_projection(self, renderer=None):
        import numpy as np
        from mpl_toolkits.mplot3d import proj3d
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

class Plot3D():
    """
    A `mpl_toolkits.mplot3d` wrapper class for plotting data in three dimensions.
    """
    def __init__(self):
        import numpy as np
        # from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import pyplot as plt
        self.xaxis = np.linspace(0, 200, 100)
        self.yaxis = np.linspace(0, 200, 100)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(29, -44)
        self.xlabel = 'x'
        self.ylabel = 'y'
        self.zlabel = 'z'
        self._title = ''
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_zlabel(self.zlabel)
        self.ax.set_title(self._title)
        self._colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
        self._numlines = 0
        self.vector_props = dict(mutation_scale = 20,  arrowstyle = '-|>',  lw=2,  shrinkA = 0,  shrinkB = 0)
    @property
    def colors(self):
        """The list of available colors for plotting lines and vectors."""
        return self._colors
    @property
    def next_color(self):
        """The color to be used for the next line or vector, if left unspecified when created."""
        return self._colors[self._numlines]
    def x_axis(self, min, max, num_samples=100):
        """
        Sets the x-axis bounds.
        
        Args:
            `min` (float):  The minimum value of the x-axis.
            `max` (float):  The maximum value of the x-axis.
            `num_samples` (int):  The number of samples to take between the minimum and maximum values.  Default is 100.
        """
        import numpy as np
        self.xaxis = np.linspace(min, max, num_samples)
    def y_axis(self, min, max, num_samples=100):
        """
        Sets the y-axis bounds.
        
        Args:
            `min` (float):  The minimum value of the y-axis.
            `max` (float):  The maximum value of the y-axis.
            `num_samples` (int):  The number of samples to take between the minimum and maximum values.  Default is 100.
        """
        import numpy as np
        self.yaxis = np.linspace(min, max, num_samples)
    def x_label(self, label):
        """
        Sets the x-axis label.
        
        Args:
            `label` (str):  The label of the x-axis.
        """
        self.xlabel = label
        self.ax.set_xlabel(label)
    def y_label(self, label):
        """
        Sets the y-axis label.
        
        Args:
            `label` (str):  The label of the y-axis.
        """
        self.ylabel = label
        self.ax.set_ylabel(label)
    def z_label(self, label):
        """
        Sets the z-axis label.
        
        Args:
            `label` (str):  The label of the z-axis.
        """
        self.zlabel = label
        self.ax.set_zlabel(label)
    def title(self, title):
        """
        Sets the title of the plot.
        
        Args:
            `title` (str):  The title of the plot.
        """
        self._title = title
        self.ax.set_title(title)

    def line(self, x, y, z=None, color=None, line_width=2):
        """
        Plots a line.
        
        Args:
            `x` (list):  The values defining the line in the x-direction.
            `y` (list):  The values defining the line in the y-direction.
            `z` (list):  The values defining the line in the z-direction.  Default is `None`, which plots a two-dimensional line.
            `color` (str):  The color of the line.  print(plot.colors) to see a list of colors.  Default is `None`, which uses the next color in the list.
            `line_width` (int):  The width of the line.  Default is 2.
        """
        if color is None: color = self._colors[self._numlines]
        if z is None: z = x
        self.ax.plot(x, y, z, color=color, linewidth=line_width)
        self._numlines += 1
        if color is None: color = self._colors[self._numlines]
        if callable(y): y = y(x)
        if z is None: self.ax.plot(x, y, color=color, linewidth=line_width)
        else: self.ax.plot(x, y, z, color=color, linewidth=line_width)
        self._numlines += 1

    def surface(self, z, colored=False):
        """
        Plots a surface.
        
        Args:
            `z` (list or function):  The values defining the surface in the z-direction, or a function that can generate them based on `x` and `y`.
            `colored` (bool):  Whether to color the surface.  Default is `False`.
        """
        import numpy as np
        from matplotlib import pyplot as plt
        xgrid, ygrid = np.meshgrid(self.xaxis, self.yaxis)
        if callable(z): z = z(xgrid, ygrid)
        if colored: self.ax.plot_surface(xgrid, ygrid, z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
        else: self.ax.plot_surface(xgrid, ygrid, z)

    def vector(self, x, y, z, alpha=1, color=None, linewidth=2, label=None):
        """
        Plots a vector.
        
        Args:
            `x` (tuple):  The x-component of the vector (x-initial, x-final).
            `y` (tuple):  The y-component of the vector (y-initial, y-final).
            `z` (tuple):  The z-component of the vector (z-initial, z-final).
            `alpha` (float):  The transparency of the vector (0-1).  Default is 1.
            `color` (str):  The color of the vector.  print(plot.colors) to see a list of colors.  Default is `None`, which uses the next color in the list.
            `linewidth` (int):  The width of the vector.  Default is 2.
            `label` (str):  The label for the vector.  Default is `None`.
        """
        if color is None: color = self._colors[self._numlines]
        self.vector_props = dict(mutation_scale = 20,  arrowstyle = '-|>', color=color,  lw=linewidth,  shrinkA = 0,  shrinkB = 0)
        a = Arrow3D(x, y, z, **self.vector_props, alpha=alpha, label = label)
        self.ax.add_artist(a)
        self._numlines += 1

    def legend(self, loc='upper right'):
        """
        Adds a legend to the plot.
        
        Args:
            `loc` (str):  The location of the legend.  Default is `upper right`.
        
        Can be any of:
            `best`, `right`, `center`\n
            `upper right`, `upper left`
            `lower right`, `lower left`\n
            `center right`, `center left`
            `lower center`, `upper center`
        """
        self.ax.legend(loc=loc)

    def show(self):
        """
        Shows the plot.
        """
        from matplotlib import pyplot as plt
        plt.show()
    
    def save(self, open=False):
        """
        Saves the plot to the current directory.
        """
        self.fig.savefig("plot3D.png")
        if open:
            import os, sys
            namespace = sys._getframe(1).f_globals                      # caller's globals
            dir_ = '\\'.join(namespace['__file__'].split('\\')[:-1])    # caller's directory up to and not including the filename
            file_path = f"{dir_}\\plot3D.png"                        # full path to the file passed as a string
            os.startfile(file_path)                                     # open the file in the default application




class Point:
    """
    Represents a point in 2D space.
    """
    def __init__(self, x=None, y=None, m=None, a=None):
        from math import sqrt, acos, degrees
        self.x = 0.0
        self.y = 100.0
        self._m = 0.0
        self._a = 0.0
        if type(x) == Point:
            self._x = x.x
            self._y = x.y
            self._m = x.m
            self._a = x.a
        elif type(x) == tuple:
            self._x = x[0]
            self._y = x[1]
            self._m = sqrt(x[0]**2 + (x[1])**2)
            self._a = 360-degrees(acos(((x[1])/(sqrt(x[0]**2 + (x[1])**2)))))
        elif not x == None and not y == None:
            self._x = x
            self._y = y
            self._m = sqrt(x**2 + (y)**2)
            self._a = 360-degrees(acos(((y)/(sqrt(x**2 + (y)**2)))))

    @property
    def x(self):
        """x coordinate of cartesian pair."""
        return self._x
    @x.setter
    def x(self, val):
        self._x = float(val)
    @property
    def y(self):
        """y coordinate of cartesian pair."""
        return self._y
    @y.setter
    def y(self, val):
        self._y = float(val)
    @property
    def m(self):
        """Magnitude of the vector defining the point."""
        return self._m
    @m.setter
    def m(self, val):
        self._m = float(val)
    @property
    def a(self):
        """Angle the vector defining the Point makes with NORTH."""
        return self._a
    @property
    def data(self):
        """The point's `x`, `y`, `m`, and `a` properties."""
        r = lambda x: "{:.3f}".format(round(x,3))
        return f"    xy: ({r(self._x)}, {r(self._y)}),  magnitude: {r(self._m)},  angle: {r(self._a)}°\n"

    def __repr__(self):
        """Return a string with the format 'Point(x, y) ||m||'"""
        return "Point({}, {}) ||{}|| {}°".format(self._x,self._y,round(self._m,3),round(self._a,3))
    def __add__(self, other):
        """Vector addition."""
        return Point(self._x + other.x, self._y + other.y)
    def __sub__(self, other):
        """Vector subtraction."""
        return Point(self._x - other.x, self._y - other.y)
    def __mul__(self, other):
        """Dot product."""
        return self._x * other.x + self._y * other.y


    
    
    
class System:
    """
    A class for interacting with the Windows OS.
    """
    def __init__(self):
        self.machine = '.'
        self.REG_PATH_1 = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System'
        self.REG_PATH_2 = r'SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Policies\System'
    
    def register_event(self, name, eventID):
        """
        Registers an Windows Application event to be used as a trigger for tasks.\n
        Can be seen in the Event Viewer, under Windows Logs -> Application, with type `Error`.

        Args:
            `name` (str):  What to name the event.
            `eventID` (int):  The ID to give the event.
        
        Returns:
            `None`
        """
        from win32evtlogutil import ReportEvent
        ReportEvent(name, eventID)
    
    def get_registry_value(reg_path, name):
        """
        Gets a value from the Windows Registry.

        Args:
            `reg_path` (str):  The path to the registry key.
            `name` (str):  The name of the value to get.
        
        Returns:
            `value` (str/int):  The registry key's value.
        """
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path, 0, winreg.KEY_READ) as registry_key:
            value = winreg.QueryValueEx(registry_key, name)[0]
            return value
    
    def set_registry_value(reg_path, name, value, type='REG_DWORD'):
        """
        Sets a value in the Windows Registry.

        Args:
            `reg_path` (str):  The path to the registry key.
            `name` (str):  The name of the value to set.
            `value` (str):  The value to set it to.
            `type` (str):  The type of the value. Default = 'REG_DWORD'.
        """
        import winreg
        if type == 'REG_SZ':
            type = winreg.REG_SZ
        elif type == 'REG_LINK':
            type = winreg.REG_LINK
        elif type == 'REG_DWORD':
            type = winreg.REG_DWORD
        elif type == 'REG_QWORD':
            type = winreg.REG_QWORD
        elif type == 'REG_BINARY':
            type = winreg.REG_BINARY
        elif type == 'REG_MULTI_SZ':
            type = winreg.REG_MULTI_SZ
        elif type == 'REG_EXPAND_SZ':
            type = winreg.REG_EXPAND_SZ
        elif type == 'REG_RESOURCE_LIST':
            type = winreg.REG_RESOURCE_LIST
        elif type == 'REG_DWORD_BIG_ENDIAN':
            type = winreg.REG_DWORD_BIG_ENDIAN
        elif type == 'REG_DWORD_LITTLE_ENDIAN':
            type = winreg.REG_DWORD_LITTLE_ENDIAN
        elif type == 'REG_QWORD_LITTLE_ENDIAN':
            type = winreg.REG_QWORD_LITTLE_ENDIAN
        elif type == 'REG_FULL_RESOURCE_DESCRIPTOR':
            type = winreg.REG_FULL_RESOURCE_DESCRIPTOR
        elif type == 'REG_RESOURCE_REQUIREMENTS_LIST':
            type = winreg.REG_RESOURCE_REQUIREMENTS_LIST
        else: raise ValueError(f"Invalid type: {type}")
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path, 0, winreg.KEY_SET_VALUE) as registry_key:
            winreg.SetValueEx(registry_key, name, 0, type, value)
    
    def user_is_admin(self):
        """
        Checks if the user is an administrator.

        Returns:
            `admin` (bool):  True if the user is an administrator, else False.
        """
        import ctypes
        try: return ctypes.windll.shell32.IsUserAnAdmin()
        except: return False
    
    def admin_privelages(self):
        """
        Checks if the User Account Control (UAC) registry values are user-level or admin-level.

        Returns:
            `admin_level` (bool):  True if UAC is admin-level, else False.
        """
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, self.REG_PATH_1, 0, winreg.KEY_READ) as registry_key:
            reg_val1 = winreg.QueryValueEx(registry_key, 'PromptOnSecureDesktop')[0]
            reg_val2 = winreg.QueryValueEx(registry_key, 'ConsentPromptBehaviorAdmin')[0]
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, self.REG_PATH_2, 0, winreg.KEY_READ) as registry_key:
            reg_val3 = winreg.QueryValueEx(registry_key, 'PromptOnSecureDesktop')[0]
            reg_val4 = winreg.QueryValueEx(registry_key, 'ConsentPromptBehaviorAdmin')[0]
        return True if reg_val1 == 0 and reg_val2 == 0 and reg_val3 == 0 and reg_val4 == 0 else False
    
    def set_privelages(self, level='admin'):
        """
        Sets the User Account Control (UAC) registry values to admin-level or user-value.

        Args:
            `level` (str):  The level to set the UAC registry values to.  Default = 'admin'.
        """
        import winreg
        if level == 'admin': value = 0
        elif level == 'user': value = 5
        else: raise ValueError(f"\n>>> {level} is not a valid level.\n")
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, self.REG_PATH_1, 0, winreg.KEY_SET_VALUE) as registry_key:
            winreg.SetValueEx(registry_key, 'PromptOnSecureDesktop', 0, winreg.REG_DWORD, value)
            winreg.SetValueEx(registry_key, 'ConsentPromptBehaviorAdmin', 0, winreg.REG_DWORD, value)
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, self.REG_PATH_2, 0, winreg.KEY_SET_VALUE) as registry_key:
            winreg.SetValueEx(registry_key, 'PromptOnSecureDesktop', 0, winreg.REG_DWORD, value)
            winreg.SetValueEx(registry_key, 'ConsentPromptBehaviorAdmin', 0, winreg.REG_DWORD, value)
    
    def run_as_admin(self, show_cmd=False):
        """
        Relaunches the calling script with administrator privelage.
        
        Args:
            `show_cmd` (bool):  If True, the command prompt window will be shown.  Default = False.
        
        Returns:
            `rc`:  The sub-process return code.
        """
        import sys
        import win32con, win32event, win32process
        from win32comext.shell import shellcon
        from win32comext.shell.shell import ShellExecuteEx
        python_exe = sys.executable
        cmdLine = [python_exe] + sys.argv
        cmd = '"%s"' % (cmdLine[0],)
        params = " ".join(['"%s"' % (x,) for x in cmdLine[1:]])
        if show_cmd: showCmd = win32con.SW_SHOWNORMAL
        else: showCmd = win32con.SW_HIDE
        if not self.admin_privelages(): self.set_privelages('admin')
        print("\nRunning elevated prompt ", params)
        printdashes(25+len(params), endtype='\n\n')
        lpVerb = 'runas'  # causes the UAC elevation prompt.
        procInfo = ShellExecuteEx(
            nShow  = showCmd,
            fMask  = shellcon.SEE_MASK_NOCLOSEPROCESS,
            lpVerb = lpVerb,
            lpFile = cmd,
            lpParameters = params
        )
        procHandle = procInfo['hProcess']
        obj = win32event.WaitForSingleObject(procHandle, win32event.INFINITE)
        rc = win32process.GetExitCodeProcess(procHandle)
        self.set_privelages('user')
        if self.admin_privelages():
            print(f"\n>>> NOTE: Admin privelages are still enabled!\n")
        return rc

    def service_running(self, service):
        """
        Checks if a service is running.
        
        Args:
            `service` (str):  The name of the service to check for.
        
        Returns:
            `running` (bool):  True if the service is running, else False.
        """
        from win32serviceutil import QueryServiceStatus
        return QueryServiceStatus(service, self.machine)[1] == 4

    def start_service(self, service):
        """
        Starts a service.
        
        Args:
            `service` (str):  The name of the service to start.
        
        Returns:
            `None`
        """
        from win32serviceutil import StartService
        running = self.service_running(service)
        if running: println(f"\n>>> Couldn't start: [{service}] is already running.")
        else:
            if not self.user_is_admin(): self.run_as_admin()
            StartService(service, self.machine)
            running = self.service_running(service)
            if running: println(f"\n>>> [{service}] started successfully.")
            else: println(f"\n>>> Couldn't start [{service}].")
    
    def stop_service(self, service):
        """
        Stops a service.
        
        Args:
            `service` (str):  The name of the service to stop.
        
        Returns:
            `None`
        """
        from win32serviceutil import StopService
        running = self.service_running(service)
        if running:
            if not self.user_is_admin(): self.run_as_admin()
            StopService(service, self.machine)
            running = self.service_running(service)
            if running: println(f"\n>>> Couldn't stop [{service}].")
            else: println(f"\n>>> [{service}] stopped successfully.")
        else: println(f"\n>>> Couldn't stop: [{service}] is not running.")
    
    def restart_service(self, service):
        """
        Restarts a service.
        
        Args:
            `service` (str):  The name of the service to restart.
        
        Returns:
            `None`
        """
        self.stop_service(service)
        self.start_service(service)




class Web3:
    """
    A wrapper for the Web3 library.  Call `print(Web3())` to see the available methods for the given contract.

    __init__ arguments:
        `contract_address` (str):  The address of the contract.
        `provider` (str):  The URL of the RPC provider. Default is Infura Polygon. Pass `"Ethereum"` to use Infura Ethereum.
    """
    def __init__(self, contract_address=None, provider=None, contract_abi=None):
        """
        Initializes the Web3 instance.

        Examples:
            >>> cntrct = Web3(contract_address='0x...', provider='Polygon')
            >>> cntrct = Web3(contract_address='0x...', provider='Ethereum')
        """
        import web3, json
        from urllib.request import urlopen

        if provider is None:
            provider = 'https://polygon-mainnet.infura.io/v3/'
            self.ABI_url = 'https://api.polygonscan.com/api?module=contract&action=getabi&address='+contract_address+'&apikey='
        elif provider == "Ethereum":
            provider = 'https://mainnet.infura.io/v3/'
            self.ABI_url = 'https://api.etherscan.io/api?module=contract&action=getabi&address='+contract_address+'&apikey='
        
        self.w3 = web3.Web3(web3.Web3.HTTPProvider(provider))
        with urlopen(self.ABI_url) as response: resp = json.dumps(json.loads(response.read().decode('utf-8'))['result'])
        resp2 = ""
        for char in resp:
            if char != '\\': resp2 += char
        resp2 = resp2[1:-1]
        self.ABI = json.loads(resp2)


        ########################################   CLEANING UP ABI, PUTTING INTO A NEW DICTIONARY   ########################################

        def GetInputsDict(abi_element):
            dict_ = {}; dict_["inputs"] = {}
            for i in range(len(abi_element["inputs"])):  dict_["inputs"][abi_element["inputs"][i]["name"]] = abi_element["inputs"][i]["type"]
            return dict_
        def GetInputsAndOutputsDict(abi_element):
            dict_ = {}; dict_["inputs"] = {}; dict_["outputs"] = {}
            for i in range(len(abi_element["inputs"])):  dict_["inputs"][abi_element["inputs"][i]["name"]] = abi_element["inputs"][i]["type"]
            for i in range(len(abi_element["outputs"])): dict_["outputs"][i+1] = abi_element["outputs"][i]["type"]
            return dict_

        self.abi_dict = {}
        self.abi_dict["constructor"] = {}
        self.abi_dict["constructor"]["inputs"] = {}
        for i in range(len(self.ABI[0]["inputs"])):
            self.abi_dict["constructor"]["inputs"][self.ABI[0]["inputs"][i]["name"]] = self.ABI[0]["inputs"][i]["type"]
        self.abi_dict["events"] = {}
        self.abi_dict["methods"] = {}
        for i in range(1,len(self.ABI)):
            if self.ABI[i]["type"] == "event":
                self.abi_dict["events"][self.ABI[i]['name']] = GetInputsDict(self.ABI[i])
            elif self.ABI[i]["type"] == "function":
                self.abi_dict["methods"][self.ABI[i]['name']] = GetInputsAndOutputsDict(self.ABI[i])

        ####################################################################################################################################


        self.contract = self.w3.eth.contract(
            address = self.w3.toChecksumAddress(contract_address),
            abi = self.ABI
        )
        self.contract_events = [
            key for key in self.contract.events.__dict__.keys()
            if key != 'abi' and key != '_events'
        ]
        self.contract_methods = [
            key for key in self.contract.functions.__dict__.keys() if
            key != 'abi' and
            key != 'web3' and
            key != 'address' and
            key != '_functions' and
            key.isupper() == False
        ]
        self.contract_methods_with_args = [str(self.contract.get_function_by_name(method)).split(' ')[1].split('>')[0] for method in self.contract_methods]

    
    def __repr__(self):
        """
        Prints the contract's methods and events.
        """
        print("\n\nContract methods:")
        print('-'*(len(max(self.contract_methods, key=len))+3))
        for method in self.contract_methods: print(f">> {method}")
        print('-'*(len(max(self.contract_methods, key=len))+3))
        print("\n\nContract events:")
        print('-'*(len(max(self.contract_methods, key=len))+3))
        for event in self.contract_events: print(f">> {event}")
        print('-'*(len(max(self.contract_methods, key=len))+3)+'\n\n')
        return ''
    
    def view_abi(self):
        """
        Calls the `view` function to view the contract's ABI with `pyjsonviewer`.
        """
        view(self.abi_dict)
    
    def view_events(self):
        """
        Calls the `view` function to view the contract's events with `pyjsonviewer`.
        """
        view(self.abi_dict["events"])

    def view_methods(self):
        """
        Calls the `view` function to view the contract's methods with `pyjsonviewer`.
        """
        view(self.abi_dict["methods"])
    
    def print_method_args(self):
        """
        Prints the contract's methods with their arguments.
        """
        print("\n\nContract methods:")
        print('-'*(len(max(self.contract_methods_with_args, key=len))+3))
        for method in self.contract_methods_with_args: print(f">> {method}")
        print('-'*(len(max(self.contract_methods_with_args, key=len))+3)+'\n\n')
        return ''

    def get_method(self, name=None):
        """
        Gets a method by name.

        Args:
            `name` (str):  The name of the method.
        
        Returns:
            `method` (web3._utils.datatypes):  The method object, which can be called directly.
        """
        if name is None: return None
        return self.contract.get_function_by_name(name)








class File:
    """
    A class to handle files and the manipulation of them.
    Pass the filename of the file object on initialization.
    """
    def __init__(self, file_name=None):
        """
        Constructor. Initializes the path to the file with the given name.
        """
        import sys
        namespace = sys._getframe(1).f_globals                      # caller's globals
        file_ = namespace['__file__'].split('\\')[-1]               # caller's filename (script name, including .py extension)
        dir_ = '\\'.join(namespace['__file__'].split('\\')[:-1])    # caller's directory up to and not including the filename
        self.file_dir  = f"{dir_}\\"                                # caller's directory
        self.file_name = None
        if file_name is not None:
            self.file_name = file_name                                  # file name
            self.file_path = f"{dir_}\\{file_name}"                     # full path to the file passed as a string
            self.file_type = file_name.split('.')[-1]                   # file type (extension)
    
    def __repr__(self) -> str:
        """
        Prints the contents of the file.
        """
        if self.file_type == 'pdf':
            from PyPDF2 import PdfFileReader
            with open(self.file_path, 'rb') as f:
                pdf = PdfFileReader(f)
                for page in pdf.pages: print(page.extractText())
        else:
            with open(self.file_path, 'r') as f: return f.read()

    def read(self, strip=True, lines=True):
        """
        Reads the file ( defaults to `readlines()`). Can return as a list of lines or a single string, depending on the `lines` argument.

        Args:
            `strip` (bool):  Whether to strip the file of newlines. Default = True.
            `lines` (bool):  Whether to return the file as a list of lines. Default = True.
        
        Returns:
            `file_contents` (list/string):  The file contents, as a list of lines or a single string.
        """
        with open(self.file_path, 'r') as f:
            if lines:
                file_contents = f.readlines()
                if strip: file_contents = [line.strip('\n') for line in file_contents]
            else:
                if strip: file_contents = f.read().strip('\n')
                else: file_contents = f.read()
        return file_contents
    
    def add(self, line):
        """
        Adds a line to the file.
        
        Args:
            `line` (str):  The line to add ('\\n' is added automatically).
        """
        with open(self.file_path, 'a') as f:
            f.write(f"{line}\n")

    def remove(self, line_content):
        """
        Removes a line from the file.

        Args:
            `line_content` (str):  The line to remove.
        """
        with open(self.file_path, 'r') as f:
            file_contents = f.readlines()
        with open(self.file_path, 'w') as f:
            for line in file_contents:
                if line.strip('\n') != line_content:
                    f.write(line)
    
    def clear(self):
        """
        Removes all lines from the file.
        """
        with open(self.file_path, 'r+') as f:
            f.truncate(0)
    
    def combine_pdf(self, file_names, output_file_name="combined.pdf"):
        """
        Combines multiple PDF files into one.
        The files are combined in the order they are passed.

        Args:
            `file_names` (list):  The list of file names to combine.
            `output_file_name` (str):  The name of the output file. Default = "combined.pdf".
        """
        from PyPDF2 import PdfFileMerger
        merger = PdfFileMerger()
        for file_name in file_names:
            if self.file_name is None:
                merger.append(self.file_dir+file_name)
            else: merger.append(file_name)
        merger.write(self.file_dir+output_file_name)
        merger.close()
        print(f"\n> Successfully wrote PDF to {self.file_dir+output_file_name}.\n")
    
    def bookmark_pdf(self, bookmark_names, page_numbers, file_name=None, output_file_name="bookmarked.pdf"):
        """
        Adds bookmarks to a PDF file.

        Args:
            `bookmark_names` (list):  A list of names for the bookmarks.
            `page_numbers` (list):  A list of page numbers corresponding to each bookmark.
            `file_name` (str):  The name of the PDF file. Will default to the file name of the File object, if provided on instantiation. Thus it can be omitted.
            `output_file_name` (str):  The name of the output file. Default = "bookmarked.pdf".
        """
        from PyPDF2 import PdfFileReader, PdfFileWriter
        writer = PdfFileWriter()
        if self.file_name is not None: reader = PdfFileReader(open(self.file_name, 'rb'))
        else: reader = PdfFileReader(open(self.file_dir+file_name, "rb"))
        for i in range(reader.getNumPages()): writer.addPage(reader.getPage(i))
        for i in range(len(bookmark_names)): writer.addBookmark(bookmark_names[i], page_numbers[i])
        with open(self.file_dir+output_file_name, 'wb') as f:
            writer.write(f)
        print(f"\n> Successfully wrote PDF to {self.file_dir+output_file_name}.\n")









class Window:
    """
    Encapsulates some calls to the winapi for window management.

    Printing an object of this class will print the window's title (the window corresponding to the object).
    """
    def __init__ (self):
        """
        Constructor.
        """
        import win32com.client
        self._handle = None
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys('%')

    def __repr__(self) -> str:
        """
        Return the window title.
        """
        import win32gui
        return str(win32gui.GetWindowText(self._handle))

    def find_window(self, class_name, window_name=None):
        """
        Find a window by its `class_name`.
        """
        import win32gui
        self._handle = win32gui.FindWindow(class_name, window_name)

    def _window_enum_callback(self, hwnd, wildcard):
        """
        Pass to `win32gui.EnumWindows` to check all the opened windows.
        """
        import re
        import win32gui
        if re.match(wildcard, str(win32gui.GetWindowText(hwnd))) is not None:
            self._handle = hwnd

    def find_window_wildcard(self, wildcard):
        """
        Find a window whose title matches the wildcard regex.
        """
        import win32gui
        self._handle = None
        win32gui.EnumWindows(self._window_enum_callback, wildcard)

    def focus(self):
        """
        Put the window in the foreground.
        """
        import win32gui
        win32gui.SetForegroundWindow(self._handle)
    
    def send_char(self, char):
        """
        Sends a character to the window.

        Args:
            `char` (str):  The character to send.
        """
        import win32api
        import win32con
        win32api.PostMessage(self._handle, win32con.WM_CHAR, char2key(char), 0)
    
    def send_string(self, string):
        """
        Sends a string to the window.
            
        Args:
            `string` (str):  The string to send.
        """
        for char in string:
            self.send_char(char)




class Keyboard:
    """
    A class that contains several methods for interacting with the keyboard.
    """
    def __init__(self):
        """
        Constructor. Instantiates the keyboard object for method usage.
        """
        from pynput.keyboard import Controller
        self._keyboard = Controller()
    
    def press(self, key):
        """
        Press a key.

        Args:
            `key` (char/string):  The key to press.
        """
        from pynput.keyboard import Key
        if key.lower() == 'space':  self._keyboard.tap(Key.space)
        elif key.lower() == 'enter':  self._keyboard.tap(Key.enter)
        elif key.lower() == 'esc':  self._keyboard.tap(Key.esc)
        elif key.lower() == 'tab':  self._keyboard.tap(Key.tab)
        elif key.lower() == 'backspace':  self._keyboard.tap(Key.backspace)
        elif key.lower() == 'up':   self._keyboard.tap(Key.up)
        elif key.lower() == 'down':   self._keyboard.tap(Key.down)
        elif key.lower() == 'left':   self._keyboard.tap(Key.left)
        elif key.lower() == 'right':  self._keyboard.tap(Key.right)
        elif key.lower() == 'lctrl':  self._keyboard.tap(Key.ctrl_l)
        elif key.lower() == 'rctrl':  self._keyboard.tap(Key.ctrl_r)
        elif key.lower() == 'lshift': self._keyboard.tap(Key.shift_l)
        elif key.lower() == 'rshift': self._keyboard.tap(Key.shift_r)
        elif key.lower() == 'lalt':   self._keyboard.tap(Key.alt_l)
        elif key.lower() == 'ralt':   self._keyboard.tap(Key.alt_r)
        elif key.lower() == 'caps':   self._keyboard.tap(Key.caps_lock)
        else:
            if key.islower():
                self._keyboard.tap(key)
            else:
                with self._keyboard.pressed(Key.shift):
                    self._keyboard.tap(key)
        
    def type(self, string):
        """
        Types a string.
        
        Args:
            `string` (string):  The string to type.
        """
        self._keyboard.type(string)

    def hold(self, key, duration):
        """
        Hold a key for a certain amount of time.

        Args:
            `key` (char/string):  The key to hold.
            `duration` (int):  The amount of time to hold the key.
        """
        from pynput.keyboard import Key
        if key.islower():
            self._keyboard.press(key)
            delay(duration)
            self._keyboard.release(key)
        else:
            with self._keyboard.pressed(Key.shift):
                self._keyboard.press(key)
                delay(duration)
                self._keyboard.release(key)

    def engage(self, key):
        """
        Engage a key.

        Args:
            `key` (char/string):  The key to engage.
        """
        self._keyboard.press(key)
    
    def disengage(self, key):
        """
        Disengage a key.

        Args:
            `key` (char/string):  The key to disengage.
        """
        self._keyboard.release(key)











class Screen:
    """
    A class that contains several methods for interacting with/obtaining information from the screen.
    """
    def __init__(self):
        """
        Constructor. Instantiates the screen object for method usage.
        """
        import win32api
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r'F:\Program Files\Tesseract-OCR\tesseract.exe'
        self.image_to_text = lambda img: pytesseract.image_to_string(img)
        self.resolution = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
        self.middle = self.resolution[0] // 2, self.resolution[1] // 2
        self.top_left = (0, 0)
        self.top_right = (self.resolution[0], 0)
        self.bottom_left = (0, self.resolution[1])
        self.bottom_right = (self.resolution[0], self.resolution[1])
    
    def get_coordinates(self, delay_time=1):
        """
        Get the xy coordinates from mouse position.

        Args:
            `delay_time` (int):  The amount of time to wait before getting the coordinates. Defaults to 1 second.
        
        Returns:
            `coordinates` (tuple):  The xy coordinates.
        """
        import mouse
        delay(delay_time)
        coordinates = mouse.get_position()
        return coordinates
    
    def pixel_color(self, x, y):
        """
        Gets the pixel color at the specified coordinates.
            
        Args:
            `x` (int):  The x coordinate.
            `y` (int):  The y coordinate.
        
        Returns:
            `color` (tuple):  The RGB color values of the pixel.
        """
        from PIL import ImageGrab
        return ImageGrab.grab().getpixel((x, y))
    
    def region(self, x1=None, y1=None, x2=None, y2=None, top_left=None, bottom_right=None):
        """
        Gets a region of the screen.

        Args:
            `x1` (int):  The x coordinate of the top left corner.
            `y1` (int):  The y coordinate of the top left corner.
            `x2` (int):  The x coordinate of the bottom right corner.
            `y2` (int):  The y coordinate of the bottom right corner.
            `top_left` (tuple):  The x, y coordinates of the top left corner. Alternative to passing as single arguments.
            `bottom_right` (tuple):  The x, y coordinates of the top right corner. Alternative to passing as single arguments.
        
        Returns:
            `region` (PIL.Image):  The region of the screen.
        """
        from PIL import ImageGrab
        if top_left and bottom_right is not None:
            x1, y1, x2, y2 = top_left[0], top_left[1], bottom_right[0], bottom_right[1]
        return ImageGrab.grab(bbox=(x1, y1, x2, y2))

    def compare(self, img1, img2):
        """
        Compares two images pixel by pixel to see if they are the same.

        Args:
            `img1` (PIL.Image):  The first image to compare.
            `img2` (PIL.Image):  The second image to compare.
        
        Returns:
            `result` (bool):  True if the images are the same, False if they are not.
        """
        from PIL import ImageChops
        return ImageChops.difference(img1, img2).getbbox() is None
        
    def extract_text(self, img):
        """
        Extracts text from an image or region from the screen (via `Screen.region`).

        Args:
            `img` (PIL.Image):  The image to extract text from.
        
        Returns:
            `text` (string):  The text from the image.
        """
        # import pytesseract
        # pytesseract.pytesseract.tesseract_cmd = r'F:\Program Files\Tesseract-OCR\tesseract.exe'
        # return pytesseract.image_to_string(img)
        return self.image_to_text(img)
    
    def read_chat(self, split_at_colon=True):
        """
        Reads the chat window from World of Warcraft.

        Args:
            `split_at_colon` (bool):  Whether to split the text at the colon, to remove the sender's name. Defaults to True.

        Returns:
            `text` (list):  The text from the chat, as a list separated by line.
        """
        chat = self.region(39,744, 553,917)
        text = self.extract_text(chat).split('\n')
        text = [line.strip().rstrip() for line in text if len(line.strip().rstrip()) > 6]
        if split_at_colon:
            text = [line.split(':',1)[1].strip().rstrip() if ':' in line else line for line in text]
        return text
        









class Math:
    """
    A class that contains math functions such as integration and differentiation.

    - Call `print(Math())` to see a list of all methods in the class.
    """
    from numpy import inf
    def __init__(self):
        """
        Initializes the Math class.
        """
        pass
    
    def __repr__(self):
        """
        Returns a string of the class's methods.
        """
        print('\n\n\n<Math> Methods:')
        methods = [m for m in self.__class__.__dict__.values() if callable(m) and m.__name__ != '__init__' and m.__name__ != '__repr__']
        for m in methods:
            print('\n'+'-'*(len(max(m.__doc__.split('\n'), key=len))))
            print('\n'+'{}()'.format(m.__name__))
            print(m.__doc__)
            print('-'*(len(max(m.__doc__.split('\n'), key=len))))
            if m == methods[-1]:  print('\n\n')
            else:  print('\n',end='')
        return ''

    def integrate(f, a=-inf, b=-inf, n=100):
        """
        Computes the integral of a function from `a` to `b` using the Composite-Trapezoidal Rule with `n` intervals.

        Args:
            `f` (function):  The function to integrate (ex: `lambda x: x**2`).
            `a` (int/float):  The lower bound of the integral. Default = -infinity.
            `b` (int/float):  The upper bound of the integral. Default = +infinity.
            `n` (int):  The number of intervals to use in the integration. Default = 100.
        
        Returns:
            `integral` (float):  The definite integral of the function.
        """
        from scipy.integrate import quad
        integral, error = quad(f, a, b, limit=n)
        return integral

    def differentiate(f, x, h=1e-5):
        """
        Computes the derivative of a function at a point `x` using the central difference method with step size `h`.

        Args:
            `f` (function):  The function to differentiate (ex: `lambda x: x**2`).
            `x` (int/float):  The point at which to differentiate.
            `h` (int/float):  The step size to use in the derivative. Default = 1e-5.
        
        Returns:
            `derivative` (float):  The derivative of the function.
        """
        from scipy.misc import derivative
        return derivative(f, x, dx=h)
        





class SolarIrradiance:
    """
    A class that contains functions for getting the current solar irradiance.
    Call `print(SolarIrradiance())` to see a list of all methods in the class.
    """
    def __init__(self):
        self.latitude  =  41.046944
        self.longitude = -95.742506
        self.API_key = "21r8H0_FfPatUB-mYgrI7cfEBSVci6_E"
    def __repr__(self):
        print('\n\n\n<SolarIrradiance> Methods:')
        print('-'*23)
        methods = [m for m in self.__class__.__dict__.values() if callable(m) and m.__name__ != '__init__' and m.__name__ != '__repr__']
        for m in methods:
            print('\n'+'-'*(len(max(m.__doc__.split('\n'), key=len))))
            print('\n'+'{}()'.format(m.__name__))
            print(m.__doc__)
            print('-'*(len(max(m.__doc__.split('\n'), key=len))))
            if m == methods[-1]:  print('\n\n\n')
            else:  print('\n',end='')
        return ''

    def current(self, latitude=None, longitude=None, returntype='dict'):
        """
        Gets the current solar irradiance, including an optional 7-day history if the `returntype` argument is set to 'list'.
        
        Args:
            `latitude` (int/float):  The latitude of the location. Default = `self.latitude` (Glenwood's latitude).
            `longitude` (int/float):  The longitude of the location. Default = `self.longitude` (Glenwood's longitude).
            `returntype` (string):  The type of data to return. Default = 'dict' (the most recent period's data).  
                                                   Can also return 'list', containing the last week's worth of data, with 30 min granularity.

        Returns:
            `current` (dict):  The current solar irradiance.
        """
        import solcast
        if latitude is None: latitude = self.latitude
        if longitude is None: longitude = self.longitude
        data = solcast.get_radiation_estimated_actuals(latitude, longitude, api_key=self.API_key)
        if data.status_code == 200:
            current_list = data.estimated_actuals
            current = data.estimated_actuals[0]
            if returntype == 'list':
                return current_list
            else:
                from datetime import datetime as dt
                # convert the datetime of the 'period_end' key to a datetime object converted to central time
                current['period_end'] = dt.strptime(current['period_end'], '%Y-%m-%dT%H:%M:%S%z')
                # convert this new datetime object to a string in the format '%m-%d-%Y %H:%M'
                current['period_end'] = current['period_end'].strftime('%m-%d-%Y %H:%M')
                return current
        else:
            print(f'\n\n>> Error:  Status code {data.status_code}\n\n')
            return None

    def forecast(self, latitude=None, longitude=None):
        """
        Gets the solar irradiance forecast for the next week.

        Args:
            `latitude` (int/float):  The latitude of the location. Default = `self.latitude` (Glenwood's latitude).
            `longitude` (int/float):  The longitude of the location. Default = `self.longitude` (Glenwood's longitude).
        
        Returns:
            `forecast` (dict):  The solar irradiance forecast.
        """
        import solcast
        if latitude is None: latitude = self.latitude
        if longitude is None: longitude = self.longitude
        data = solcast.get_radiation_forecasts(latitude, longitude, api_key=self.API_key)
        if data.status_code == 200:
            forecast = data.forecasts
            for i in range(len(data.forecasts)):
                forecast[i]['period_end'] = convert_to_local_time(forecast[i]['period_end'], fmt='%m-%d-%Y %H:%M')
            # create a new dictionary with the 'period_end' key of each element in the list as the new key
            forecast = {f['period_end']: f for f in forecast}
            # remove the 'period' and 'period_end' keys from each element in the dictionary
            for k, v in forecast.items():
                del v['period']
                del v['period_end']
            return forecast
        else:
            print(f'\n\n>> Error:  Status code {data.status_code}\n\n')
            return None





class Git:
    """
    A class that contains git functions for adding, comitting, and pushing.
    Primarily used for pushing to Heroku.

    - Call `print(Git())` to see a list of all methods in the class.
    """

    def __init__(self, email=None, password=None):
        """
        Initializes the Git class.
        """
        import os
        import time
        self.sleep = time.sleep
        self.system = os.system
        self.email = email
        self.password = password

    def __repr__(self):
        """
        Returns a string of the class's methods.
        """
        print('\n\n\n<Git> Methods:')
        methods = [m for m in self.__class__.__dict__.values() if callable(m) and m.__name__ != '__init__' and m.__name__ != '__repr__']
        for m in methods:
            print('\n'+'-'*(len(max(m.__doc__.split('\n'), key=len))))
            print('\n'+'{}()'.format(m.__name__))
            print(m.__doc__)
            print('-'*(len(max(m.__doc__.split('\n'), key=len))))
            if m == methods[-1]:  print('\n\n')
            else:  print('\n',end='')
        return ''
    
    def login(self):
        """
        Logs in to Heroku using manual method.
        """
        a = self.system('heroku whoami')
        if a != 0:
            self.system('heroku login -i')
        else:
            print('Already logged in. Passing...')
        
    def add(self, file=None):
        """
        Adds a file to git.

        Args:
            `file` (str):  The file to add. Default = None (adds all files).
        
        Returns:
            `None`
        """
        if file is None:  self.system('git add .')
        else:  self.system('git add {}'.format(file))
    
    def commit(self, message='No message'):
        """
        Commits the current git repository.

        Args:
            `message` (str):  The commit message. Default = 'No message'.
        
        Returns:
            `None`
        """
        self.system('git commit -m "{}"'.format(message))
    
    def push(self):
        """
        Pushes the current git repository to Heroku.

        Args:
            `None`
        
        Returns:
            `None`
        """
        self.system('git push heroku master')
    
    def heroku(self):
        """
        Function to handle adding, committing, and pushing to Heroku.
        """
        self.login()
        self.sleep(1)
        self.add()
        self.sleep(1)
        self.commit()
        self.sleep(1)
        self.push()







class Regex:
    """
    A class for simplifying the matching of things in strings via RegEx patterns.

    - Call `print(Regex())` to see a summary of regex stuff.

    """
    # https://www.programiz.com/python-programming/regex
    def __init__(self, search_string=None):
        """
        Constructor. Takes the string to be searched, and defines the special sequences.
        """
        self.search_string = search_string
        self.start = r'\A'
        self.end = r'\Z'
        self.boundary = r'\b'
        self.non_boundary = r'\B'
        self.digit = r'\d'
        self.non_digit = r'\D'
        self.whitespace = r'\s'
        self.non_whitespace = r'\S'
        self.alphanumeric = r'\w'
        self.non_alphanumeric = r'\W'
    
    def __repr__(self):
        """
        Prints a summary of the regex stuff.
        """
        print("\n\n"+
            REGEX_1+REGEX_2+REGEX_3+REGEX_4+REGEX_5+REGEX_6+REGEX_7+REGEX_8+REGEX_9+REGEX_10+
            REGEX_11+REGEX_12+REGEX_13+REGEX_14+REGEX_15+REGEX_16+REGEX_17+REGEX_18+REGEX_19+
            REGEX_20+REGEX_21+REGEX_22+REGEX_23+REGEX_24+REGEX_25+REGEX_26+REGEX_27+REGEX_28+
            "\n\n")
        return ''
    
    def find_any_characters(self, length):
        """
        Finds any single character in the string.

        Args:
            `length` (int):  The number of characters to qualify as a match.
            
            Example:
                `length = 2`, `search_string = ab`:  Match
                `length = 2`, `search_string = a`:    No match

        Returns:
            `match` (list):  A list of strings containing the matched character(s).
        """
        import re
        pattern = r'.' * length
        return re.findall(pattern, self.search_string)
    
    def find_specific_characters(self, characters, invert=False):
        """
        Searches for a set of characters in the search string.

        Args:
            `characters` (str):  The characters to search for.
            `invert` (bool):  If True, will search for characters that are not in the string. Default is False.
        
        Returns:
            `match` (list):  A list of strings where the characters were found.
        """
        import re
        if invert: pattern = '[^' + characters + ']'
        else: pattern = '[' + characters + ']'
        match = re.findall(pattern, self.search_string)
        return match
    
    def find_this_or_that(self, this, that, followed_by=None):
        """
        Searches for an instance of one character or another appearing in the search string.

        Args:
            `this` (str):  The first character to search for.
            `that` (str):  The other character to search for.
            `followed_by` (str):  The character(s) that must follow either character. Default is None.
        
        Returns:
            `match` (list):  A list of strings where the characters were found.
        """
        import re
        if followed_by: pattern = '(' + this + '|' + that + ')' + followed_by
        else: pattern = this + '|' + that
        match = re.findall(pattern, self.search_string)
        return match






REGEX_1  = "\n[] - specifies a set of characters you wish to match\n\t"+"[abc] (will match if the string you are trying to match contains any of the a, b, or c characters) ([^abc] matches all characters except a, b, or c) ([^0-9] matches all characters except numbers)\n\t\t"+"a          -   1 match\n\t\t"+"ac         -   2 matches\n\t\t"+"Hey Jude   -   0 matches\n\t\t"+"abc de ca  -   5 matches\n\n"
REGEX_2  = "\n.  - matches any single character (except newline '\\n')\n\t"+"..\n\t\t"+"a    -   No match\n\t\t"+"ac   -   1 match\n\t\t"+"acd  -   1 match\n\t\t"+"acde -   2 matches (contains 4 characters)\n\n"
REGEX_3  = "\n^  - used to check if a string starts with a certain character"+"\n\t^a"+"\n\t\ta   -   1 match"+"\n\t\tabc -   1 match"+"\n\t\tbac -   No match"+"\n\t^ab"+"\n\t\tabc -   1 match"+"\n\t\tacb -   No match (starts with a, but not followed by b)\n\n"
REGEX_4  = "\n$  - used to check if a string ends with a certain character"+"\n\ta$"+"\n\t\ta\t-   1 match"+"\n\t\tformula -   1 match"+"\n\t\tcab\t-   No match\n\n"
REGEX_5  = "\n*  - matches zero or more occurrences of the pattern left to it"+"\n\tma*n"+"\n\t\tmn     -   1 match"+"\n\t\tman    -   1 match"+"\n\t\tmaaan  -   1 match"+"\n\t\tmain   -   No match (a is not followed by n)"+"\n\t\twoman  -   1 match\n\n"
REGEX_6  = "\n+  - matches one or more occurrences of the pattern left to it"+"\n\tma+n"+"\n\t\tmn    -   No match (no a character)"+"\n\t\tman   -   1 match"+"\n\t\tmaaan -   1 match"+"\n\t\tmain  -   No match (a is not followed by n)"+"\n\t\twoman -   1 match\n\n"
REGEX_7  = "\n?  - matches zero or one occurrences of the pattern left to it"+"\n\tma?n"+"\n\t\tmn    -   1 match"+"\n\t\tman   -   1 match"+"\n\t\tmaaan -   No match (more than one a character)"+"\n\t\tmain  -   No match (a is not followed by n)"+"\n\t\twoman -   1 match\n\n"
REGEX_8  = "\n{} - ex: {n, m} - matches at least n, and at most m occurrences of the pattern left to it"+"\n\ta{2,3}"+"\n\t\tabc dat\t-    No match"+"\n\t\tabc daat    -    1 match (at daat)"+"\n\t\t\t\t\t\t\t   ^^"+"\n\t\taabc daaat  -  2 matches (at aabc and daaat)"+"\n\t\t\t\t\t\t\t    ^^\t   ^^^"+"\n\t\taabc daaaat -  2 matches (at aabc and daaaat)"+"\n\t\t\t\t\t\t\t    ^^\t   ^^^"+"\n\t [0-9]{2,4}"+"\n\t\tab123csde\t  -   1 match (at ab123csde)"+"\n\t\t\t\t\t\t\t\t^^^"+"\n\t\t12 and 345673   -   3 matches (at 12, 3456, and 73)"+"\n\t\t\t\t\t\t\t\t    ^^  ^^^^\t ^^"+"\n\t\t1 and 2\t    -   No match\n\n"
REGEX_9  = "\n|  - used for alteration (or operator)"+"\n\ta|b (match any string containing either a or b)"+"\n\t\tcde\t-   No match"+"\n\t\tade\t-   1 match (at ade)"+"\n\t\t\t\t\t\t    ^"+"\n\t\tacdbea  -   3 matches (at acdbea)"+"\n\t\t\t\t\t\t\t ^  ^ ^\n\n"
REGEX_10  = "\n() - used for grouping sub-patterns (Ex: (a|b|c)xz matches any string that matches either a or b or c followed by xz)"+"\n\t(a|b|c)xz"+"\n\t\tab xz\t -   No match"+"\n\t\tabxz\t  -   1 match (at abxz)"+"\n\t\t\t\t\t\t\t^^^"+"\n\t\taxz cabxz  -  2 matches (at axzbc and cabxz)"+"\n\t\t\t\t\t\t\t  ^^^\t    ^^^\n\n"
REGEX_11  = "\n\  - used to escape various characters including all metacharacters"+"\n\t(Ex: \$a matches if a string contains $ followed by a (Note: $ is not interpreted as a metacharacter by the RegEx engine here))\n\n"
REGEX_12  = "\n\n\nSpecial Sequences\n\n"
REGEX_13  = "\n\A  - matches the beginning of the string"+"\n\t \Athe"+"\n\t\t    the sun    -   match"+"\n\t\t    In the sun -   No match\n\n"
REGEX_14  = "\n\\\b  - matches a word boundary (beginning or end of a word)"+"\n\t \bfoo"+"\n\t\t    football -   match"+"\n\t\t    a football - match"+"\n\t\t    afootball  - No match"+"\n\t foo\\b"+"\n\t\t    the foo\t    -  match"+"\n\t\t    the afoot test  -  match"+"\n\t\t    the afootest    -  No match\n\n"
REGEX_15  = "\n\B  - opposite of \\b. matches anything but a word boundary (if the specified characters are NOT at the beginning or end of a word)"+"\n\t \Bfoo"+"\n\t\t    football -   No match"+"\n\t\t    a football - No match"+"\n\t\t    afootball  - match"+"\n\t foo\B"+"\n\t\t    the foo\t    -  No match"+"\n\t\t    the afoot test  -  No match"+"\n\t\t    the afootest    -  match\n\n"
REGEX_16  = "\n\d  - matches any decimal digit (equivalent to [0-9])"+"\n\t \d"+"\n\t\t    12abc3   -   3 matches (at 12abc3)"+"\n\t\t\t\t\t\t\t\t ^^   ^"+"\n\t\t    Python   -   No match\n\n"
REGEX_17  = "\n\D  - matches any non-decimal digit (equivalent to [^0-9])"+"\n\t \D"+"\n\t\t    1ab34\"50   -   3 matches (at 1ab34\"50)"+"\n\t\t\t\t\t\t\t\t    ^^  ^"+"\n\t\t    1345\t  -   No match\n\n"
REGEX_18  = "\n\s  - matches where a string contains any whitespace character (space, tab, newline, etc.) (equivalent to [ \\f\\n\\r\\t\\v])"+"\n\t \s"+"\n\t\t    Python RegEx   -  1 match"+"\n\t\t    PythonRegEx    -  No match\n\n"
REGEX_19  = "\n\S  - matches where a string contains any non-whitespace character (equivalent to [^ \\f\"\\n\\r\\t\\v])"+"\n\t \S"+"\n\t\t    a b   -  2 matches (at a b)"+"\n\t\t\t\t\t\t\t  ^ ^"+"\n\t\t    ' '   -  No match\n\n"
REGEX_20  = "\n\w  - matches any alphanumeric character (digits and alphabets) (equivalent to [a-zA-Z0-9_])"+"\n\t \w"+"\n\t\t    12&\": ;c  -  3 matches (at 12&\": ;c)"+"\n\t\t\t\t\t\t\t\t ^^\t^"+"\n\t\t    %\"> !\t-  No match\n\n"
REGEX_21  = "\n\W  - matches any non-alphanumeric character (equivalent to [^a-zA-Z0-9_])"+"\n\t \W"+"\n\t\t    1a2%c\t-  1 match (at 1a2%c)"+"\n\t\t\t\t\t\t\t\t^ ^"+"\n\t\t    Python    -  No match\n\n"
REGEX_22  = "\n\Z  - matches the end of the string"+"\n\t Python\Z"+"\n\t\t    I like Python\t\t\t-   1 match"+"\n\t\t    I like Python Programming   -   No match"+"\n\t\t    Python is fun.\t\t    -   No match\n\n"
REGEX_23  = "\nre.findall()  - returns a list of strings containing all matches\n"
REGEX_24  = "re.split()    - splits the string where there is a match and returns a list of strings where the splits have occurred"+"\n\t\t\t if pattern not found, returns a list containing the original string"+"\n\t\t\t you can pass `maxsplit` to limit the number of splits that will occur (default = 0, which means no limit)\n"
REGEX_25  = "re.sub()      - returns a string where matched occurrences are replaced with the content of `replace` variable."+"\n\t\t\t if the pattern is not found, returns the original string"+"\n\t\t\t you can pass `count` as a fourth parameter. if omitted, it defaults to 0 (this will replace all occurrences)\n"
REGEX_26  = "re.subn()     - similar to `re.sub()`, except it returns a tuple of 2 items containing the new string and the number of substitutions made\n"
REGEX_27  = "re.search()   - looks for the first location where the RegEx pattern produces a match with the string"+"\n\t\t\t if the search is successful, it returns a Match object, otherwise it returns None\n\n"
REGEX_28  = "Match Object (you can get methods and attributes of a match object using dir() function)\n"+" .group()\t- returns the part of the string where there is a match\n"+" .start()\t- returns the index of the start of the matched substring\n"+" .end()\t\t- returns the end index of the matched substring\n"+" .span()\t- returns a tuple containing the start and end index of the matched substring\n"+" .re\t\t- returns the regular expression object that was used to search\n"+" .string\t- returns the string that was searched/passed in\n"
