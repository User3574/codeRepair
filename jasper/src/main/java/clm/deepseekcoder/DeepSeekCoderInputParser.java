package clm.deepseekcoder;

import com.github.javaparser.Range;
import com.github.javaparser.ast.Node;

import clm.jasper.ContextParser;
import clm.jasper.Parser;

public class DeepSeekCoderInputParser {
    
    /**
     * Given the buggy file, the buggy line, and an input config, get the codet5 input
     * 
     * @param filename the buggy file
     * @param startLine the start line of the buggy lines
     * @param endLine the endline of the buggy lines (included)
     * @param config a config in enum CodeT5Config
     * @return a CodeT5Input object
     */
    public static DeepSeekCoderInput getDeepSeekCoderInput(String filename, int startLine, int endLine, DeepSeekCoderConfig config) {
        try {
            Node buggyFunctionNode = ContextParser.getSurroundingFunctionNode(filename, startLine, endLine, true);
            Range range = buggyFunctionNode.getRange().get();
            String functionRange = range.begin.line + "," + range.begin.column + "-" + range.end.line + "," + range.end.column;
            String input = "";
            if (config == DeepSeekCoderConfig.DEEPSEEKCODER_COMPLETE_CODEFORM_NOCOMMENT) {
                String buggyFunctionBefore = ContextParser.getSurroundingFunctionBefore(filename, startLine, endLine, true);
                String buggyFunctionAfter = ContextParser.getSurroundingFunctionAfter(filename, startLine, endLine, true);
                int buggyLineIndent = Parser.getIndent(filename, startLine);
                int buggyFunctionIndent = Parser.getIndent(buggyFunctionNode);

                input = "<｜fim▁begin｜>" + buggyFunctionBefore + "\n";
                // add the idnentation
                for (int i = 0; i < buggyLineIndent - buggyFunctionIndent; i += 1)
                    input += " ";
                input += "<｜fim▁hole｜>\n" + buggyFunctionAfter + "<｜fim▁end｜>";
            } else if (config == DeepSeekCoderConfig.DEEPSEEKCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT) {
                String buggyFunctionBefore = ContextParser.getSurroundingFunctionBefore(filename, startLine, endLine, true);
                String buggyFunctionAfter = ContextParser.getSurroundingFunctionAfter(filename, startLine, endLine, true);
                String buggyLine = ContextParser.getDedentedCode(filename, startLine, endLine, true);
                int buggyLineIndent = Parser.getIndent(filename, startLine);
                int buggyFunctionIndent = Parser.getIndent(buggyFunctionNode);

                input = "<｜fim▁begin｜>" + buggyFunctionBefore;
                for (String line : buggyLine.split("\\n")) {
                    if (line.trim().equals(""))
                        continue;
                    input += "\n// buggy line:" + line;
                }
                input += "\n";
                // add the idnentation
                for (int i = 0; i < buggyLineIndent - buggyFunctionIndent; i += 1)
                    input += " ";
                input += "<｜fim▁hole｜>\n" + buggyFunctionAfter + "<｜fim▁end｜>";
            }
            input = Parser.removeEmptyLines(input);
            return new DeepSeekCoderInput(input, functionRange);
        } catch (Exception e){
            System.out.println(e);
            return new DeepSeekCoderInput("", "");
        }
    }

    /**
     * Dump the codet5 result into a json file
     * 
     * @param filename the buggy file
     * @param startLine the start line of the buggy lines
     * @param endLine the endline of the buggy lines (included)
     * @param config a config in enum DeepSeekCoderConfig
     * @param outputFileName the json file to dump the codex input
     */
    public static void dumpDeepSeekCoderInput(String filename, int startLine, int endLine, DeepSeekCoderConfig config, String outputFileName) throws Exception{
        DeepSeekCoderInput deepseekcoderInput = getDeepSeekCoderInput(filename, startLine, endLine, config);
        deepseekcoderInput.dumpAsJson(outputFileName);
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
            DeepSeekCoderConfig config = null;
            if (args[3].equals("DEEPSEEKCODER_COMPLETE_CODEFORM_NOCOMMENT")){
                config = DeepSeekCoderConfig.DEEPSEEKCODER_COMPLETE_CODEFORM_NOCOMMENT;
            } else if (args[3].equals("DEEPSEEKCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT")) {
                config = DeepSeekCoderConfig.DEEPSEEKCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT;
            } else {
                throw new Exception("Unrecognized DeepSeekCoderConfig: " + args[3]);
            }
            String outputFileName = args[4];
            dumpDeepSeekCoderInput(filename, startLine, endLine, config, outputFileName);
        } else {
            throw new Exception("Arguments number mismatched, expected 5, but got " + args.length);
        }
    }
}
