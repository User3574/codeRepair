package clm.bloom;

import com.github.javaparser.Range;
import com.github.javaparser.ast.Node;

import clm.jasper.ContextParser;
import clm.jasper.Parser;

public class BloomInputParser {
    
    /**
     * Given the buggy file, the buggy line, and an input config, get the bloom input
     * 
     * @param filename the buggy file
     * @param startLine the start line of the buggy lines
     * @param endLine the endline of the buggy lines (included)
     * @param config a config in enum Bloom
     * @return a BloomInput object
     */
    public static BloomInput getBloomInput(String filename, int startLine, int endLine, BloomConfig config) {
        try {
            Node buggyFunctionNode = ContextParser.getSurroundingFunctionNode(filename, startLine, endLine, true);
            Range range = buggyFunctionNode.getRange().get();
            String functionRange = range.begin.line + "," + range.begin.column + "-" + range.end.line + "," + range.end.column;
            String input = "";
            if (config == BloomConfig.BLOOM_COMPLETE_CODEFORM_NOCOMMENT) {
                String buggyFunctionBefore = ContextParser.getSurroundingFunctionBefore(filename, startLine, endLine, true);
                input = buggyFunctionBefore;
            } else if (config == BloomConfig.BLOOM_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT) {
                String buggyFunctionBefore = ContextParser.getSurroundingFunctionBefore(filename, startLine, endLine, true);
                String buggyLine = ContextParser.getDedentedCode(filename, startLine, endLine, true);
                input = buggyFunctionBefore;
                for (String line : buggyLine.split("\\n")) {
                    if (line.trim().equals(""))
                        continue;
                    input += "\n// buggy line:" + line;
                }
            }
            input = Parser.removeEmptyLines(input);
            return new BloomInput(input, functionRange);
        } catch (Exception e){
            System.out.println(e);
            return new BloomInput("", "");
        }
    }

    /**
     * Dump the codetgen result into a json file
     * 
     * @param filename the buggy file
     * @param startLine the start line of the buggy lines
     * @param endLine the endline of the buggy lines (included)
     * @param config a config in enum BloomConfig
     * @param outputFileName the json file to dump the codex input
     */
    public static void dumpBloomInput(String filename, int startLine, int endLine, BloomConfig config, String outputFileName) throws Exception {
        BloomInput bloomInput = getBloomInput(filename, startLine, endLine, config);
        bloomInput.dumpAsJson(outputFileName);
    }

    
    /** 
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length == 5) {
            String filename = args[0];
            int startLine = Integer.parseInt(args[1]);
            int endLine = Integer.parseInt(args[2]);
            BloomConfig config = null;
            if (args[3].equals("BLOOM_COMPLETE_CODEFORM_NOCOMMENT")){
                config = BloomConfig.BLOOM_COMPLETE_CODEFORM_NOCOMMENT;
            } else if (args[3].equals("BLOOM_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT")) {
                config = BloomConfig.BLOOM_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT;
            } else {
                throw new Exception("Unrecognized BloomConfig: " + args[3]);
            }
            String outputFileName = args[4];
            dumpBloomInput(filename, startLine, endLine, config, outputFileName);
        } else {
            throw new Exception("Arguments number mismatched, expected 5, but got " + args.length);
        }
    }

}
