# NANDmaster - The chess problem solver

**Instructions for running the program:**
1. Activate Python Virtual Environment in the 'venv' folder
2. Position command line in 'src' folder
3. Run main.py with desired arguments

**Supported arguments:**
- arg1 - input PNG file path
- arg2 - player turn ['white' | 'black']
- optional arg3 - chess AI max think time in seconds [>0]
- optional arg4 - opponent AI skill level [0 - 20]

**Examples:**<br>
<code>python main.py chess_problems/mate_in_4_0.png white</code><br>
<code>python main.py chess_problems/mate_in_4_0.png white 2.5 0</code>
