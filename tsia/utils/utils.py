def hex_to_rgb(hex_color):
    """
    Converts a color string in hexadecimal format to RGB format.
    
    PARAMS
    ======
        hex_color: string
            A string describing the color to convert from hexadecimal. It can
            include the leading # character or not
    
    RETURNS
    =======
        rgb_color: tuple
            Each color component of the returned tuple will be a float value
            between 0.0 and 1.0
    """
    hex_color = hex_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], base=16) / 255.0 for i in [0, 2, 4])
    return rgb_color