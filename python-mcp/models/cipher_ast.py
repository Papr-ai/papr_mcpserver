from enum import Enum
from typing import List, Union, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from models.structured_outputs import (
    NodeLabel, RelationshipType,
    InsightProperties, MeetingProperties, 
    TaskProperties, OpportunityProperties,
    CodeProperties, MemoryProperties, 
    PersonProperties, CompanyProperties,
    ProjectProperties
)
import json
import logging

logger = logging.getLogger(__name__)


class NodeAlias(str, Enum):
    SOURCE = 'm'
    TARGET = 'n'
    RELATIONSHIP = 'r'

class Direction(str, Enum):
    BOTH = "-"

class ComparisonOperator(str, Enum):
    # Equality
    EQUALS = "="
    NOT_EQUALS = "<>"
    
    # Numeric comparisons
    GREATER_THAN = ">"
    GREATER_THAN_EQUALS = ">="
    LESS_THAN = "<"
    LESS_THAN_EQUALS = "<="
    
    # Pattern matching
    CONTAINS = "CONTAINS"
    STARTS_WITH = "STARTS WITH"
    ENDS_WITH = "ENDS WITH"
    
    # Collection operations
    IN = "IN"
    NOT_IN = "NOT IN"
    
    # Null checks
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    
    # Regular expression
    MATCHES = "=~"

# Map NodeLabel to their corresponding property types
NODE_PROPERTY_MAP = {
    NodeLabel.Insight: InsightProperties,
    NodeLabel.Meeting: MeetingProperties,
    NodeLabel.Task: TaskProperties,
    NodeLabel.Opportunity: OpportunityProperties,
    NodeLabel.Code: CodeProperties,
    NodeLabel.Memory: MemoryProperties,
    NodeLabel.Person: PersonProperties,
    NodeLabel.Company: CompanyProperties,
    NodeLabel.Project: ProjectProperties
}

# First, collect all property names at module level
ALL_NODE_PROPERTIES = tuple(
    prop for node_type in NODE_PROPERTY_MAP.values() 
    for prop in node_type.model_fields.keys()
)

class WhereCondition(BaseModel):
    """Represents a WHERE condition in Cypher"""
    # Create a Union of all possible property names from all node types
    property: Union[Literal[*ALL_NODE_PROPERTIES]]
    operator: ComparisonOperator
    value: Union[str, int, float, bool, list]
    and_operator: Union[bool, None] = True
    node_label: Optional[NodeLabel] = None

    @field_validator('value')
    @classmethod
    def validate_value(cls, v: Any) -> Any:
        """Validate that string values don't contain special characters"""
        if isinstance(v, str):
            if any(char in v for char in '{},[]'):
                raise ValueError(f"String values cannot contain special characters like {{}}, [], or ,")
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str) and any(char in item for char in '{},[]'):
                    raise ValueError("List string values cannot contain special characters like {}, [], or ,")
        return v

    @field_validator('property')
    @classmethod
    def validate_property_exists(cls, v: str, info) -> str:
        # Get the node label from context if available
        node_label = info.data.get('node_label')
        if node_label and node_label in NODE_PROPERTY_MAP:
            property_model = NODE_PROPERTY_MAP[node_label]
            valid_properties = property_model.model_fields.keys()
            if v not in valid_properties:
                raise ValueError(
                    f"Property '{v}' is not valid for node type {node_label}. "
                    f"Valid properties are: {valid_properties}"
                )
        return v

    def to_cypher(self) -> str:
        """Convert condition to Cypher string with proper quoting"""
        # Get the operator value directly
        operator_value = self.operator.value
        
        # Handle different value types
        if isinstance(self.value, str):
            value_str = f"'{self.value}'"
        elif isinstance(self.value, list):
            value_items = [f"'{item}'" if isinstance(item, str) else str(item) for item in self.value]
            value_str = f"[{', '.join(value_items)}]"
        elif self.operator in [ComparisonOperator.IS_NULL, ComparisonOperator.IS_NOT_NULL]:
            value_str = ""
        else:
            value_str = str(self.value)
        
        # Handle special cases for IS NULL/IS NOT NULL
        if self.operator in [ComparisonOperator.IS_NULL, ComparisonOperator.IS_NOT_NULL]:
            return f"{self.property} {operator_value}"
        
        return f"{self.property} {operator_value} {value_str}"

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "property": {
                    "anyOf": [
                        {
                            "type": "string",
                            "enum": list(model.model_fields.keys())
                        } for model in NODE_PROPERTY_MAP.values()
                    ]
                },
                "operator": {
                    "type": "string",
                    "enum": [op.value for op in ComparisonOperator]
                },
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "null"},
                        {"type": "array"}
                    ]
                },
                "and_operator": {
                    "type": ["boolean", "null"],
                    "default": True
                },
                "node_label": {
                    "type": ["string", "null"],
                    "enum": [label.value for label in NodeLabel]
                }
            },
            "required": ["property", "operator", "value", "and_operator"]
        }
    )

class CipherNode(BaseModel):
    alias: str
    label: NodeLabel
    conditions: Optional[List[WhereCondition]] = None

    @field_validator('conditions')
    @classmethod
    def validate_conditions(cls, v, values):
        if v is None:
            return v
        
        # Get the node label
        node_label = values.data.get('label')
        if node_label and node_label in NODE_PROPERTY_MAP:
            property_model = NODE_PROPERTY_MAP[node_label]
            valid_properties = property_model.model_fields.keys()
            
            for condition in v:
                if condition.property not in valid_properties:
                    raise ValueError(
                        f"Property '{condition.property}' in condition is not valid for node type {node_label}. "
                        f"Valid properties are: {valid_properties}"
                    )
        return v

    def to_cypher(self) -> str:
        """Convert node to Cypher syntax"""
        node_str = f"({self.alias}:{self.label.value})"
        if self.conditions:
            where_conditions = [cond.to_cypher() for cond in self.conditions]
            node_str += f" WHERE {' AND '.join(where_conditions)}"
        return node_str

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "alias": {
                    "type": "string",
                    "enum": [e.value for e in NodeAlias]
                },
                "label": {
                    "type": "string",
                    "enum": [l.value for l in NodeLabel]
                },
                "conditions": {
                    "type": ["array", "null"],
                    "default": None,
                    "items": {
                        "type": "object",
                        "properties": {
                            "property": {
                                "anyOf": [
                                    {
                                        "type": "string",
                                        "enum": list(model.model_fields.keys())
                                    } for model in NODE_PROPERTY_MAP.values()
                                ]
                            },
                            "operator": {
                                "type": "string",
                                "enum": [op.value for op in ComparisonOperator]
                            },
                            "value": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "number"},
                                    {"type": "boolean"},
                                    {"type": "null"},
                                    {"type": "array"}
                                ]
                            }
                        },
                        "required": ["property", "operator", "value"]
                    }
                }
            },
            "required": ["alias", "label"]
        }
    )

class Edge(BaseModel):
    relationship: RelationshipType
    direction: Direction
    conditions: Optional[List[WhereCondition]] = None
    alias: str

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "relationship": {
                    "type": "string",
                    "enum": [r.value for r in RelationshipType]
                },
                "direction": {
                    "type": "string",
                    "enum": [d.value for d in Direction]
                },
                "conditions": {
                    "type": ["array", "null"],
                    "default": None,
                    "items": {
                        "type": "object",
                        "properties": {
                            "property": {"type": "string"},
                            "operator": {
                                "type": "string",
                                "enum": [op.value for op in ComparisonOperator]
                            },
                            "value": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "number"},
                                    {"type": "boolean"},
                                    {"type": "array"}
                                ]
                            }
                        },
                        "required": ["property", "operator", "value"]
                    }
                },
                "alias": {
                    "type": "string",
                    "enum": [NodeAlias.RELATIONSHIP.value]
                }
            },
            "required": ["relationship", "direction", "alias"]
        }
    )

class PatternElement(BaseModel):
    left_node: CipherNode
    relationship: Edge
    right_node: CipherNode

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "left_node": {
                    "$ref": "#/$defs/CipherNode"
                },
                "relationship": {
                    "$ref": "#/$defs/Edge"
                },
                "right_node": {
                    "$ref": "#/$defs/CipherNode"
                }
            },
            "required": ["left_node", "relationship", "right_node"],
            "additionalProperties": False,
            "$defs": {
                "CipherNode": {
                    "type": "object",
                    "properties": {
                        "alias": {
                            "type": "string",
                            "enum": [e.value for e in NodeAlias]
                        },
                        "label": {
                            "type": "string",
                            "enum": [l.value for l in NodeLabel]
                        },
                        "conditions": {
                            "type": "array",
                            "items": {
                                "$ref": "#/$defs/WhereCondition"
                            }
                        }
                    },
                    "required": ["alias", "label"]
                },
                "Edge": {
                    "$ref": "#/model_config/json_schema_extra/$defs/Edge"
                },
                "WhereCondition": {
                    "$ref": "#/model_config/json_schema_extra/$defs/WhereCondition"
                }
            }
        }
    )
    
    def to_cypher(self) -> str:
        # Build the base pattern
        left_part = f"({self.left_node.alias}:{self.left_node.label.value})"
        rel_part = f"[{self.relationship.alias}:{self.relationship.relationship.value}]"
        right_part = f"({self.right_node.alias}:{self.right_node.label.value})"
        
        pattern = f"{left_part}-{rel_part}-{right_part}"
        
        # Add WHERE conditions if they exist
        where_conditions = []
        
        # Process conditions for both nodes
        for node, conditions in [
            (self.right_node, self.right_node.conditions),
            (self.left_node, self.left_node.conditions)
        ]:
            if conditions:
                for i, cond in enumerate(conditions):
                    if isinstance(cond.value, str):
                        value = f"'{cond.value}'"
                    elif isinstance(cond.value, list):
                        value_items = [f"'{item}'" if isinstance(item, str) else str(item) for item in cond.value]
                        value = f"[{', '.join(value_items)}]"
                    else:
                        value = str(cond.value)
                    
                    condition = f"{node.alias}.{cond.property} {cond.operator.value} {value}"
                    
                    # Only add condition if it has a value
                    if cond.value is not None:
                        # First condition doesn't need an operator
                        if not where_conditions:
                            where_conditions.append(condition)
                        else:
                            # Subsequent conditions must have an operator (AND/OR)
                            if cond.and_operator is True:
                                where_conditions.append("AND")
                            elif cond.and_operator is False:
                                where_conditions.append("OR")
                            else:
                                # If no operator specified, skip this condition
                                continue
                            where_conditions.append(condition)

        # Combine pattern with WHERE clause if conditions exist
        if where_conditions:
            conditions_str = ' '.join(where_conditions)
            return f"{pattern}\nWHERE {conditions_str}"
        return pattern

class MatchClause(BaseModel):
    pattern: PatternElement

    def to_cypher(self) -> str:
        """Convert match clause to Cypher string"""
        return f"MATCH {self.pattern.to_cypher()}"

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "pattern": {
                    "$ref": "#/$defs/PatternElement"
                }
            },
            "required": ["pattern"],
            "additionalProperties": False,
            "$defs": {
                "PatternElement": {
                    "$ref": "#/model_config/json_schema_extra/$defs/PatternElement"
                },
                "CipherNode": {
                    "$ref": "#/model_config/json_schema_extra/$defs/CipherNode"
                },
                "Edge": {
                    "$ref": "#/model_config/json_schema_extra/$defs/Edge"
                },
                "WhereCondition": {
                    "$ref": "#/model_config/json_schema_extra/$defs/WhereCondition"
                }
            }
        }
    )

class ReturnClause(BaseModel):
    expressions: List[NodeAlias]
    order_by: Union[str, None]
    aggregation: Union[str, None]

    @field_validator('expressions')
    @classmethod
    def validate_expressions(cls, v):
        required_aliases = [NodeAlias.SOURCE, NodeAlias.RELATIONSHIP, NodeAlias.TARGET]
        
        # Check if all required aliases are present
        missing_aliases = [alias for alias in required_aliases if alias not in v]
        if missing_aliases:
            raise ValueError(f"Return clause must include all variables: {required_aliases}. Missing: {missing_aliases}")
        
        # Check if there are any invalid aliases
        invalid_aliases = [expr for expr in v if expr not in required_aliases]
        if invalid_aliases:
            raise ValueError(f"Invalid return expressions: {invalid_aliases}. Must use: {required_aliases}")
        
        # Ensure the order is always m, r, n
        return required_aliases

    def to_cypher(self) -> str:
        """Convert return clause to Cypher with enforced spacing."""
        return_expr = [alias.value for alias in self.expressions]
        if self.aggregation:
            return_expr = [f"{self.aggregation}({expr})" for expr in return_expr]
        return "RETURN " + ", ".join(return_expr)  # Enforced space after RETURN

class CypherQuery(BaseModel):
    match: MatchClause

    @field_validator('match')
    @classmethod
    def validate_pattern(cls, v: MatchClause) -> MatchClause:
        """Validate the single pattern and its conditions"""
        # Validate string values in conditions for both nodes
        for node in [v.pattern.left_node, v.pattern.right_node]:
            if node.conditions:
                for condition in node.conditions:
                    if isinstance(condition.value, str):
                        if any(char in condition.value for char in '{},[]'):
                            raise ValueError(f"Invalid characters in condition value: {condition.value}")
        return v
    
    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "type": "object",
            "properties": {
                "match": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "object",
                            "properties": {
                                "left_node": {
                                    "type": "object",
                                    "properties": {
                                        "alias": {
                                            "type": "string",
                                            "enum": [str(e.value) for e in NodeAlias]
                                        },
                                        "label": {
                                            "type": "string",
                                            "enum": [str(l.value) for l in NodeLabel]
                                        },
                                        "conditions": {
                                            "type": ["array", "null"],
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "property": {
                                                        "anyOf": [
                                                            {
                                                                "type": "string",
                                                                "enum": list(model.model_fields.keys())
                                                            } for model in NODE_PROPERTY_MAP.values()
                                                        ]
                                                    },
                                                    "operator": {
                                                        "type": "string",
                                                        "enum": [str(op.value) for op in ComparisonOperator]
                                                    },
                                                    "value": {
                                                        "anyOf": [
                                                            {"type": "string"},
                                                            {"type": "number"},
                                                            {"type": "boolean"},
                                                            {"type": "null"},
                                                            {
                                                                "type": "array",
                                                                "items": {
                                                                    "anyOf": [
                                                                        {"type": "string"},
                                                                        {"type": "number"},
                                                                        {"type": "boolean"}
                                                                    ]
                                                                }
                                                            }
                                                        ]
                                                    },
                                                    "and_operator": {
                                                        "type": ["boolean", "null"]
                                                    }
                                                },
                                                "required": ["property", "operator", "value", "and_operator"]
                                            }
                                        }
                                    },
                                    "required": ["alias", "label", "conditions"]
                                },
                                "relationship": {
                                    "type": "object",
                                    "properties": {
                                        "relationship": {
                                            "type": "string",
                                            "enum": [str(r.value) for r in RelationshipType]
                                        },
                                        "direction": {
                                            "type": "string",
                                            "enum": [str(d.value) for d in Direction]
                                        },
                                        "alias": {
                                            "type": "string",
                                            "enum": [str(NodeAlias.RELATIONSHIP.value)]
                                        }
                                    },
                                    "required": ["relationship", "direction", "alias"]
                                },
                                "right_node": {
                                    "type": "object",
                                    "properties": {
                                        "alias": {
                                            "type": "string",
                                            "enum": [str(e.value) for e in NodeAlias]
                                        },
                                        "label": {
                                            "type": "string",
                                            "enum": [str(l.value) for l in NodeLabel]
                                        },
                                        "conditions": {
                                            "type": ["array", "null"],
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "property": {
                                                        "anyOf": [
                                                            {
                                                                "type": "string",
                                                                "description": f"Properties for {label.value}",
                                                                "enum": list(NODE_PROPERTY_MAP[label].model_fields.keys())
                                                            } for label in NodeLabel
                                                        ]
                                                    },
                                                    "operator": {
                                                        "type": "string",
                                                        "enum": [str(op.value) for op in ComparisonOperator]
                                                    },
                                                    "value": {
                                                        "anyOf": [
                                                            {"type": "string"},
                                                            {"type": "number"},
                                                            {"type": "boolean"},
                                                            {"type": "null"},
                                                            {
                                                                "type": "array",
                                                                "items": {
                                                                    "anyOf": [
                                                                        {"type": "string"},
                                                                        {"type": "number"},
                                                                        {"type": "boolean"}
                                                                    ]
                                                                }
                                                            }
                                                        ]
                                                    }
                                                },
                                                "required": ["property", "operator", "value"]
                                            }
                                        }
                                    },
                                    "required": ["alias", "label", "conditions"]
                                }
                            },
                            "required": ["left_node", "relationship", "right_node"]
                        }
                    },
                    "required": ["pattern"]
                }
            },
            "required": ["match"]
        }
    )
    
    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        """Override to add logging for schema generation"""
        schema = super().model_json_schema()
        logger.debug(f"Generated JSON Schema for CypherQuery: {json.dumps(schema, indent=2)}")
        return schema

    def to_cypher(self) -> str:
        """Convert the entire query to Cypher string"""
        # Get the base query from match clause
        query = self.match.to_cypher()
        
        # No need to replace operator values anymore since we handle them in WhereCondition
        return query