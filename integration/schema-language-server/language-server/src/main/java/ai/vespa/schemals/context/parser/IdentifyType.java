package ai.vespa.schemals.context.parser;

import java.io.PrintStream;
import java.util.ArrayList;

import org.eclipse.lsp4j.Diagnostic;
import org.eclipse.lsp4j.DiagnosticSeverity;
import org.eclipse.lsp4j.Range;

import com.yahoo.schema.parser.ParsedType;
import com.yahoo.schema.parser.ParsedType.Variant;
import ai.vespa.schemals.parser.Token.TokenType;
import ai.vespa.schemals.tree.SchemaNode;
import ai.vespa.schemals.tree.TypeNode;

public class IdentifyType extends Identifier {


    public IdentifyType(PrintStream logger) {
        super(logger);
    }

    private boolean isPrimitiveType(SchemaNode node) {
        ParsedType type = ParsedType.fromName(node.getText());
        return type.getVariant() != Variant.UNKNOWN;
    }

    private ArrayList<Diagnostic> validateType(SchemaNode node) {
        ArrayList<Diagnostic> ret = new ArrayList<Diagnostic>();

        ParsedType type = ParsedType.fromName(node.getText());
        if (type.getVariant() == Variant.UNKNOWN) {
            ret.add(new Diagnostic(node.getRange(), "Invalid type"));
        } else {
            new TypeNode(node);
        }

        return ret;
    }

    public ArrayList<Diagnostic> identify(SchemaNode node) {
        ArrayList<Diagnostic> ret = new ArrayList<Diagnostic>();

        SchemaNode parent = node.getParent();
        // TODO: handle types in inputs and constants
        if (node.getType() != TokenType.TYPE 
            || parent == null 
            || parent.indexOf(node) == -1 
            || parent.indexOf(node) + 1 == parent.size())return ret;

        int childIndex = parent.indexOf(node) + 1;

        SchemaNode child = parent.get(childIndex);
        // Check if it uses deprecated array
        if (
            child.getClassLeafIdentifierString().equals("dataType") &&
            child.size() > 1 &&
            child.get(1).getText().equals("[]")
        ) {
            Range range = child.getRange();

            child = child.get(0);

            ret.add(new Diagnostic(range, "Data type syntax '" + child.getText() + "[]' is deprecated, use 'array<" + child.getText() + ">' instead.", DiagnosticSeverity.Warning,""));
        }

        if (isPrimitiveType(child)) {
            return ret;
        }

        Range range = child.getRange();
        ret.add(new Diagnostic(range, ParsedType.fromName(child.getText()).toString(), DiagnosticSeverity.Error, ""));

        // Check if type is valid
        //if (
        //    child.size() > 2 &&
        //    child.get(1).getType() == TokenType.LESSTHAN
        //) {
        //    TokenType firstChildType = child.get(0).getType();
        //    if (
        //        firstChildType != TokenType.ANNOTATIONREFERENCE &&
        //        firstChildType != TokenType.REFERENCE
        //    ) {
        //        for (int i = 2; i < child.size(); i += 2) {
        //            ret.addAll(validateType(child.get(i)));
        //        }
        //    }
        //} else if (child.getType() != TokenType.TENSOR_TYPE) {
        //    ret.addAll(validateType(child));
        //    child.setType(null);
        //}

        return ret;
    }
}
