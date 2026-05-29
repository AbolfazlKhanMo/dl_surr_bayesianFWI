
def modify_par_file(file_path, line_number, new_content):
    # Read the contents of the file into memory
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Modify the specific line
    if 1 <= line_number <= len(lines):
        lines[line_number - 1] = new_content + '\n'  # Adjust line_number to zero-based index

        # Write the modified contents back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)
        # print(f"Line {line_number} modified successfully.")
    else:
        print("Invalid line number. No changes were made.")



# EXAMPLE USAGE
# file_path = 'Par_file_MCMC_DATA'  # Change to your file path
# line_number = 287                 # Params for physical group 1
# proposed_vp = 3500
# new_content = f'1 1 2250 {proposed_vp} 2010 0 0 9999 9999 0 0 0 0 0 0'

# modify_par_file(file_path, line_number, new_content)



