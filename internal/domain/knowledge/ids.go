package knowledge

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
)

func stableHash(parts ...string) string {
	sum := sha256.Sum256([]byte(strings.Join(parts, "|")))
	return hex.EncodeToString(sum[:])
}

func CanonicalEntityID(entityType EntityType, normalizedName string) string {
	return stableHash(string(NormalizeEntityType(entityType)), normalizedName)
}

func MentionID(docHash string, chunkID int, pgID int64, entityID, normalizedSurface string) string {
	return stableHash(docHash, strconv.Itoa(chunkID), strconv.FormatInt(pgID, 10), entityID, normalizedSurface)
}

func RelationID(fromEntityID, toEntityID, relType, docHash string, chunkID int, pgID int64) string {
	return stableHash(fromEntityID, toEntityID, NormalizeRelationType(relType), docHash,
		strconv.Itoa(chunkID), strconv.FormatInt(pgID, 10))
}

func KGChunkID(pgID int64) string { return fmt.Sprintf("pg:%d", pgID) }
