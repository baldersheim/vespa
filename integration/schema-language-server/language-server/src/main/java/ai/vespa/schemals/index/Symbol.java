package ai.vespa.schemals.index;

import ai.vespa.schemals.common.FileUtils;
import ai.vespa.schemals.tree.SchemaNode;

import org.eclipse.lsp4j.Location;
import org.eclipse.lsp4j.Position;

public class Symbol {
    private SchemaNode identifierNode;
    private Symbol scope = null;
    private String fileURI;
    private SymbolType type;
    private SymbolStatus status;
    private String shortIdentifier;

    public Symbol(SchemaNode identifierNode, SymbolType type, String fileURI, Symbol scope, String shortIdentifier) {
        this.identifierNode = identifierNode;
        this.fileURI = fileURI;
        this.type = type;
        this.status = SymbolStatus.UNRESOLVED;
        this.scope = scope;
        this.shortIdentifier = shortIdentifier;
    }

    public Symbol(SchemaNode identifierNode, SymbolType type, String fileURI) {
        this(identifierNode, type, fileURI, null, identifierNode.getText());
    }

    public Symbol(SchemaNode identifierNode, SymbolType type, String fileURI, Symbol scope) {
        this(identifierNode, type, fileURI, scope, identifierNode.getText());
    }

    public String getFileURI() { return fileURI; }
    
    public String setFileURI(String fileURI) {
        this.fileURI = fileURI;
        return fileURI;
    }
    
    public void setType(SymbolType type) { this.type = type; }
    public SymbolType getType() { return type; }
    public void setStatus(SymbolStatus status) { this.status = status; }
    public SymbolStatus getStatus() { return status; }

    public Symbol getScope() { return scope; }

    // TODO: not quite sure if this kind of equality check is good
    public boolean isInScope(Symbol scope) {
        if (scope == null || this.scope == null) return false;
        return this.scope.equals(scope);
    }

    public String getScopeIdentifier() {
        if (this.scope == null) return "";
        return this.scope.getLongIdentifier();
    }

    public SchemaNode getNode() { return identifierNode; }

    public String getShortIdentifier() { return shortIdentifier; }

    public String getLongIdentifier() {
        if (scope == null) {
            return getShortIdentifier();
        }
        return scope.getLongIdentifier() + "." + getShortIdentifier();
    }

    public Location getLocation() {
        return new Location(fileURI, identifierNode.getRange());
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }
        Symbol other = (Symbol) obj;
        return (
            this.fileURI.equals(other.fileURI) &&
            this.type == other.type &&
            this.status == other.status &&
            this.getNode() != null &&
            other.getNode() != null &&
            this.getNode().getRange() != null &&
            other.getNode().getRange() != null &&
            this.getNode().getRange().equals(other.getNode().getRange())
        );
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((this.fileURI == null) ? 0 : this.fileURI.hashCode());
        result = prime * result + ((this.type == null) ? 0 : this.type.hashCode());
        result = prime * result + ((this.getNode() == null || this.getNode().getRange() == null) ? 0 : this.getNode().getRange().hashCode());
        return result;
    }

    public enum SymbolStatus {
        DEFINITION,
        REFERENCE,
        UNRESOLVED,
        INVALID,
        BUILTIN_REFERENCE // reference to stuff like "default" that doesn't have a definition in our CSTs
    }

    public enum SymbolType {
        SCHEMA,
        DOCUMENT,
        FIELD,
        STRUCT,
        ANNOTATION,
        RANK_PROFILE,
        FIELDSET,
        STRUCT_FIELD,
        FUNCTION,
        DOCUMENT_SUMMARY,
        SUMMARY,
        TYPE_UNKNOWN,
        SUBFIELD,
        MAP_KEY,
        MAP_VALUE,
        PARAMETER
    }

    public String toString() {
        Position pos = getNode().getRange().getStart();
        String fileName = FileUtils.fileNameFromPath(fileURI);
        return "Symbol('" + getShortIdentifier() + "', scope: '" + getScopeIdentifier() + "', at: " + fileName + ":" + pos.getLine() + ":" + pos.getCharacter() + ")@" + System.identityHashCode(this);
    }
}
