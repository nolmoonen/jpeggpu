# TODO

- Implement full baseline. With the restriction of only supporting the popular subsamplings and 8 bits data (8 bits qtable?). TODO:
  - Using the correct Huffman tables, possibly four instead of two
- Separate copy functions in API interface.
- Check if CSS is one of "popular subsamplings", reject otherwise
- Add subsamplings and color formats to convert function and interface
- Add some test.
- Add documentation in this file: what it does, references to papers, some benchmark results
- Can do some refactors and renames guided by TODOs.
- Improve logger, TODO:
  - Log all CUDA errors through the logger.
  - Make it so logger does not need to be passed around.
  - Log allocation sizes.
