package clm.starcoder;

import com.github.javaparser.Range;
import com.github.javaparser.ast.Node;

import clm.jasper.ContextParser;
import clm.jasper.Parser;

public class StarCoderInputParser {
    
    /**
     * Given the buggy file, the buggy line, and an input config, get the codet5 input
     * 
     * @param filename the buggy file
     * @param startLine the start line of the buggy lines
     * @param endLine the endline of the buggy lines (included)
     * @param config a config in enum CodeT5Config
     * @return a CodeT5Input object
     */
    public static StarCoderInput getStarCoderInput(String filename, int startLine, int endLine, StarCoderConfig config) {
        try {
            Node buggyFunctionNode = ContextParser.getSurroundingFunctionNode(filename, startLine, endLine, true);
            Range range = buggyFunctionNode.getRange().get();
            String functionRange = range.begin.line + "," + range.begin.column + "-" + range.end.line + "," + range.end.column;
            String input = "";
            if (config == StarCoderConfig.STARCODER_COMPLETE_CODEFORM_NOCOMMENT) {
                String buggyFunctionBefore = ContextParser.getSurroundingFunctionBefore(filename, startLine, endLine, true);
                String buggyFunctionAfter = ContextParser.getSurroundingFunctionAfter(filename, startLine, endLine, true);
                int buggyLineIndent = Parser.getIndent(filename, startLine);
                int buggyFunctionIndent = Parser.getIndent(buggyFunctionNode);

                input = "<fim_prefix>" + buggyFunctionBefore + "\n";
                // add the idnentation
                for (int i = 0; i < buggyLineIndent - buggyFunctionIndent; i += 1)
                    input += " ";
                input += "<fim_suffix>\n" + buggyFunctionAfter + "<fim_middle>";
            } else if (config == StarCoderConfig.STARCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT) {
                String buggyFunctionBefore = ContextParser.getSurroundingFunctionBefore(filename, startLine, endLine, true);
                String buggyFunctionAfter = ContextParser.getSurroundingFunctionAfter(filename, startLine, endLine, true);
                String buggyLine = ContextParser.getDedentedCode(filename, startLine, endLine, true);
                int buggyLineIndent = Parser.getIndent(filename, startLine);
                int buggyFunctionIndent = Parser.getIndent(buggyFunctionNode);

                input = "<fim_prefix>" + buggyFunctionBefore;
                for (String line : buggyLine.split("\\n")) {
                    if (line.trim().equals(""))
                        continue;
                    input += "\n// buggy line:" + line;
                }
                input += "\n";
                // add the idnentation
                for (int i = 0; i < buggyLineIndent - buggyFunctionIndent; i += 1)
                    input += " ";
                input += "<fim_suffix>\n" + buggyFunctionAfter + "<fim_middle>";
            }
            input = Parser.removeEmptyLines(input);
            return new StarCoderInput(input, functionRange);
        } catch (Exception e){
            System.out.println(e);
            return new StarCoderInput("", "");
        }
    }

    /**
     * Dump the codet5 result into a json file
     * 
     * @param filename the buggy file
     * @param startLine the start line of the buggy lines
     * @param endLine the endline of the buggy lines (included)
     * @param config a config in enum StarCoderConfig
     * @param outputFileName the json file to dump the codex input
     */
    public static void dumpStarCoderInput(String filename, int startLine, int endLine, StarCoderConfig config, String outputFileName) throws Exception{
        StarCoderInput starcoderInput = getStarCoderInput(filename, startLine, endLine, config);
        starcoderInput.dumpAsJson(outputFileName);
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
            StarCoderConfig config = null;
            if (args[3].equals("STARCODER_COMPLETE_CODEFORM_NOCOMMENT")){
                config = StarCoderConfig.STARCODER_COMPLETE_CODEFORM_NOCOMMENT;
            } else if (args[3].equals("STARCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT")) {
                config = StarCoderConfig.STARCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT;
            } else {
                throw new Exception("Unrecognized StarCoderConfig: " + args[3]);
            }
            String outputFileName = args[4];
            dumpStarCoderInput(filename, startLine, endLine, config, outputFileName);
        } else {
            throw new Exception("Arguments number mismatched, expected 5, but got " + args.length);
        }
    }
}
